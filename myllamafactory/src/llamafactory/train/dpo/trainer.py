# Copyright 2024 HuggingFace Inc. and the LlamaFactory team.
#
# This code is inspired by the HuggingFace's TRL library.
# https://github.com/huggingface/trl/blob/v0.8.0/trl/trainer/dpo_trainer.py
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import warnings
from collections import defaultdict
from contextlib import nullcontext
from types import MethodType
from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from transformers import Trainer
# from .mpo_trainer import MPOTrainer
from trl.trainer import DPOTrainer
from trl.trainer import disable_dropout_in_model

from ...extras.constants import IGNORE_INDEX
from ..trainer_utils import convert_pissa_adapter, create_custom_optimzer, create_custom_scheduler, get_batch_logps


if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments


class CustomDPOTrainer(DPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        if disable_dropout:
            disable_dropout_in_model(model)
            if ref_model is not None:
                disable_dropout_in_model(ref_model)

        self.finetuning_args = finetuning_args
        self.processor = processor
        self.reference_free = False
        self.use_dpo_data_collator = True  # hack to avoid warning
        self.generate_during_eval = False  # disable at evaluation
        self.label_pad_token_id = IGNORE_INDEX
        self.padding_value = 0
        self.is_encoder_decoder = model.config.is_encoder_decoder
        self.precompute_ref_log_probs = False
        self._precomputed_train_ref_log_probs = False
        self._precomputed_eval_ref_log_probs = False
        self._peft_has_been_casted_to_bf16 = False

        self.ref_model = ref_model
        self._stored_metrics = defaultdict(lambda: defaultdict(list))

        # dpo hyperparams
        self.beta = finetuning_args.pref_beta
        self.loss_type = finetuning_args.pref_loss
        self.ftx_gamma = finetuning_args.pref_ftx
        self.have_w = finetuning_args.have_w
        self.have_extra = finetuning_args.have_extra
        self.p_x = finetuning_args.p_x
        self.n_x = 1-finetuning_args.p_x
        self.dgx_gamma = finetuning_args.pref_dgx
        self.label_smoothing = finetuning_args.dpo_label_smoothing
        self.simpo_gamma = finetuning_args.simpo_gamma

        Trainer.__init__(self, model=model, **kwargs)
        if not hasattr(self, "accelerator"):
            raise AttributeError("Please update `transformers`.")

        warnings.simplefilter("ignore")  # remove gc warnings on ref model

        if ref_model is not None:
            if self.is_deepspeed_enabled:
                if not (
                    getattr(ref_model, "is_loaded_in_8bit", False) or getattr(ref_model, "is_loaded_in_4bit", False)
                ):  # quantized models are already set on the correct device
                    self.ref_model = self._prepare_deepspeed(self.ref_model)
            else:
                self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
                self.ref_model.eval()

        if finetuning_args.pissa_convert:
            self.save_model(os.path.join(self.args.output_dir, "pissa_init"))

        if finetuning_args.use_badam:
            from badam import clip_grad_norm_for_sparse_tensor

            self.accelerator.clip_grad_norm_ = MethodType(clip_grad_norm_for_sparse_tensor, self.accelerator)

    def create_optimizer(self) -> "torch.optim.Optimizer":
        if self.optimizer is None:
            self.optimizer = create_custom_optimzer(self.model, self.args, self.finetuning_args)
        return super().create_optimizer()

    def create_scheduler(
        self, num_training_steps: int, optimizer: Optional["torch.optim.Optimizer"] = None
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(self.args, num_training_steps, optimizer)
        return super().create_scheduler(num_training_steps, optimizer)

    def _save(self, output_dir: Optional[str] = None, state_dict: Optional[Dict[str, "torch.Tensor"]] = None) -> None:
        super()._save(output_dir, state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        if self.finetuning_args.pissa_convert:
            convert_pissa_adapter(output_dir, state_dict, self.accelerator, self.model, self.args)

        if self.processor is not None:
            getattr(self.processor, "image_processor").save_pretrained(output_dir)

    def odds_ratio_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes ORPO's odds ratio (OR) loss for batched log probabilities of the policy model.
        """
        log_odds = (chosen_logps - rejected_logps) - (
            torch.log1p(-torch.exp(chosen_logps)) - torch.log1p(-torch.exp(rejected_logps))
        )
        sft_loss = -chosen_logps
        odds_ratio_loss = -F.logsigmoid(log_odds)
        orpo_loss = sft_loss + self.beta * odds_ratio_loss
        return orpo_loss

    def simpo_loss(self, chosen_logps: "torch.Tensor", rejected_logps: "torch.Tensor") -> "torch.Tensor":
        r"""
        Computes SimPO loss for batched log probabilities of the policy model.
        """
        pi_logratios = chosen_logps - rejected_logps
        gamma_logratios = self.simpo_gamma / self.beta
        logits = pi_logratios - gamma_logratios
        simpo_loss = -F.logsigmoid(self.beta * logits)
        return simpo_loss

    def compute_preference_loss(
        self,
        policy_chosen_logps: "torch.Tensor",
        policy_rejected_logps: "torch.Tensor",
        reference_chosen_logps: Optional["torch.Tensor"],
        reference_rejected_logps: Optional["torch.Tensor"],
        p_policy_chosen_logps: "torch.Tensor",
        p_policy_rejected_logps: "torch.Tensor",
        p_reference_chosen_logps: Optional["torch.Tensor"],
        p_reference_rejected_logps: Optional["torch.Tensor"],
        n_policy_chosen_logps: "torch.Tensor",
        n_policy_rejected_logps: "torch.Tensor",
        n_reference_chosen_logps: Optional["torch.Tensor"],
        n_reference_rejected_logps: Optional["torch.Tensor"],
    ):
        r"""
        Computes loss for preference learning.
        """
        if not self.finetuning_args.use_ref_model:
            if self.loss_type == "orpo":
                losses = self.odds_ratio_loss(policy_chosen_logps, policy_rejected_logps)
                p_losses = self.odds_ratio_loss(p_policy_chosen_logps, p_policy_rejected_logps)
                n_losses = self.odds_ratio_loss(n_policy_chosen_logps, n_policy_rejected_logps)
            elif self.loss_type == "simpo":
                losses = self.simpo_loss(policy_chosen_logps, policy_rejected_logps)
                p_losses = self.simpo_loss(p_policy_chosen_logps, p_policy_rejected_logps)
                n_losses = self.simpo_loss(n_policy_chosen_logps, n_policy_rejected_logps)
            else:
                raise NotImplementedError("Unknown loss type: {}.".format(self.loss_type))

            chosen_rewards = self.beta * policy_chosen_logps.to(self.accelerator.device).detach()
            rejected_rewards = self.beta * policy_rejected_logps.to(self.accelerator.device).detach()
            if self.have_extra:
                p_chosen_rewards = self.beta * p_policy_chosen_logps.to(self.accelerator.device).detach()
                p_rejected_rewards = self.beta * p_policy_rejected_logps.to(self.accelerator.device).detach()
                n_chosen_rewards = self.beta * n_policy_chosen_logps.to(self.accelerator.device).detach()
                n_rejected_rewards = self.beta * n_policy_rejected_logps.to(self.accelerator.device).detach()
                losses = (losses+self.dgx_gamma*p_losses+self.dgx_gamma*n_losses)
                return losses, chosen_rewards, rejected_rewards, p_chosen_rewards, p_rejected_rewards, n_chosen_rewards, n_rejected_rewards
            else:
                return losses, chosen_rewards, rejected_rewards
        else:
            losses, chosen_rewards, rejected_rewards= self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            if self.have_extra:
                p_losses, p_chosen_rewards, p_rejected_rewards= self.dpo_loss(
                    p_policy_chosen_logps, p_policy_rejected_logps, p_reference_chosen_logps, p_reference_rejected_logps
                )
                n_losses, n_chosen_rewards, n_rejected_rewards= self.dpo_loss(
                    n_policy_chosen_logps, n_policy_rejected_logps, n_reference_chosen_logps, n_reference_rejected_logps
                )
                losses = (losses+self.dgx_gamma*p_losses+self.dgx_gamma*n_losses)
                return losses, chosen_rewards, rejected_rewards, p_chosen_rewards, p_rejected_rewards, n_chosen_rewards, n_rejected_rewards
            else:
                return losses, chosen_rewards, rejected_rewards
    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor","torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor","torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor", "torch.Tensor"]:
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

        # 获取批次大小
        batch_size = batch["input_ids"].size(0)
        third_size = batch_size // 3


        if self.have_extra:
            # 将 batch 分为三部分
            batch1 = {k: v[:third_size] for k, v in batch.items()}
            all_logits: "torch.Tensor" = model(**batch1, return_dict=True, use_cache=False).logits.to(torch.float32)
            all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch1["labels"])

            batch2 = {k: v[third_size:2*third_size] for k, v in batch.items()}
            p_all_logits: "torch.Tensor" = model(**batch2, return_dict=True, use_cache=False).logits.to(torch.float32)
            p_all_logps, p_valid_length = get_batch_logps(logits=p_all_logits, labels=batch2["labels"])

            batch3 = {k: v[2*third_size:] for k, v in batch.items()}
            n_all_logits: "torch.Tensor" = model(**batch3, return_dict=True, use_cache=False).logits.to(torch.float32)
            n_all_logps, n_valid_length = get_batch_logps(logits=n_all_logits, labels=batch3["labels"])

            if self.loss_type in ["ipo", "orpo", "simpo"]:
                all_logps = all_logps / valid_length
                p_all_logps = p_all_logps / p_valid_length
                n_all_logps = n_all_logps / n_valid_length

            batch_size = batch1["input_ids"].size(0) // 2
            chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
            chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
            chosen_length, rejected_length = valid_length.split(batch_size, dim=0)
            
            p_chosen_logps, p_rejected_logps = p_all_logps.split(batch_size, dim=0)
            p_chosen_logits, p_rejected_logits = p_all_logits.split(batch_size, dim=0)
            p_chosen_length, _ = p_valid_length.split(batch_size, dim=0)
            
            n_chosen_logps, n_rejected_logps = n_all_logps.split(batch_size, dim=0)
            n_chosen_logits, n_rejected_logits = n_all_logits.split(batch_size, dim=0)
            n_chosen_length, _ = n_valid_length.split(batch_size, dim=0)
            
            if self.loss_type in ["ipo", "orpo", "simpo"]:
                policy_chosen_logps_avg = chosen_logps*valid_length/chosen_length
                policy_rejected_logps_avg = rejected_logps*valid_length/rejected_length
                p_policy_chosen_logps_avg = p_chosen_logps*p_valid_length/p_chosen_length
                n_policy_chosen_logps_avg = n_chosen_logps*n_valid_length/n_chosen_length
            else:
                policy_chosen_logps_avg = chosen_logps/chosen_length
                policy_rejected_logps_avg = rejected_logps/rejected_length
                p_policy_chosen_logps_avg = p_chosen_logps/p_chosen_length
                n_policy_chosen_logps_avg = n_chosen_logps/n_chosen_length
            # p_p
            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, policy_chosen_logps_avg, policy_rejected_logps_avg, p_chosen_logps, p_rejected_logps, p_chosen_logits, p_rejected_logits, p_policy_chosen_logps_avg, n_chosen_logps, n_rejected_logps, n_chosen_logits, n_rejected_logits, n_policy_chosen_logps_avg
        else:
            # 将 batch 分为三部分
            batch1 = {k: v[:third_size] for k, v in batch.items()}
            all_logits: "torch.Tensor" = model(**batch1, return_dict=True, use_cache=False).logits.to(torch.float32)
            all_logps, valid_length = get_batch_logps(logits=all_logits, labels=batch1["labels"])

            if self.loss_type in ["ipo", "orpo", "simpo"]:
                all_logps = all_logps / valid_length

            batch_size = batch1["input_ids"].size(0) // 2
            chosen_logps, rejected_logps = all_logps.split(batch_size, dim=0)
            chosen_logits, rejected_logits = all_logits.split(batch_size, dim=0)
            chosen_length, rejected_length = valid_length.split(batch_size, dim=0)
            
            if self.loss_type in ["ipo", "orpo", "simpo"]:
                policy_chosen_logps_avg = chosen_logps*valid_length/chosen_length
                policy_rejected_logps_avg = rejected_logps*valid_length/rejected_length
            else:
                policy_chosen_logps_avg = chosen_logps/chosen_length
                policy_rejected_logps_avg = rejected_logps/rejected_length

            return chosen_logps, rejected_logps, chosen_logits, rejected_logits, policy_chosen_logps_avg, policy_rejected_logps_avg, None, None, None, None, None, None, None, None, None, None
 
    def compute_reference_log_probs(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ) -> Tuple[Optional["torch.Tensor"], Optional["torch.Tensor"],Optional["torch.Tensor"], Optional["torch.Tensor"],Optional["torch.Tensor"], Optional["torch.Tensor"]]:
        r"""
        Computes log probabilities of the reference model.
        """
        if not self.finetuning_args.use_ref_model:
            return None, None

        if self.ref_model is None:
            ref_model = model
            ref_context = self.accelerator.unwrap_model(model).disable_adapter()
        else:
            ref_model = self.ref_model
            ref_context = nullcontext()

        with torch.no_grad(), ref_context:
            reference_chosen_logps, reference_rejected_logps, _,_,_,_, p_reference_chosen_logps, p_reference_rejected_logps, _,_,_, n_reference_chosen_logps, n_reference_rejected_logps, _,_,_ = self.concatenated_forward(ref_model, batch)

        return reference_chosen_logps, reference_rejected_logps, p_reference_chosen_logps, p_reference_rejected_logps, n_reference_chosen_logps, n_reference_rejected_logps

    def get_batch_loss_metrics(
        self,
        model: "PreTrainedModel",
        batch: Dict[str, "torch.Tensor"],
        train_eval: Literal["train", "eval"] = "train",
    ) -> Tuple["torch.Tensor", Dict[str, "torch.Tensor"]]:
        r"""
        Computes the DPO loss and other metrics for the given batch of inputs for train or test.
        """
        metrics = {}
        (
            policy_chosen_logps,
            policy_rejected_logps,
            policy_chosen_logits,
            policy_rejected_logits,
            policy_chosen_logps_avg,
            policy_rejected_logps_avg,
            p_policy_chosen_logps,
            p_policy_rejected_logps,
            p_policy_chosen_logits,
            p_policy_rejected_logits,
            p_policy_chosen_logps_avg,
            n_policy_chosen_logps,
            n_policy_rejected_logps,
            n_policy_chosen_logits,
            n_policy_rejected_logits,
            n_policy_chosen_logps_avg,
        ) = self.concatenated_forward(model, batch)

        reference_chosen_logps, reference_rejected_logps, p_reference_chosen_logps, p_reference_rejected_logps, n_reference_chosen_logps, n_reference_rejected_logps = self.compute_reference_log_probs(model, batch)
        if self.have_extra:
            losses, chosen_rewards, rejected_rewards, p_chosen_rewards, p_rejected_rewards, n_chosen_rewards, n_rejected_rewards = self.compute_preference_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
            p_policy_chosen_logps,
            p_policy_rejected_logps,
            p_reference_chosen_logps,
            p_reference_rejected_logps,
            n_policy_chosen_logps,
            n_policy_rejected_logps,
            n_reference_chosen_logps,
            n_reference_rejected_logps,
        )
            sft_loss = -policy_chosen_logps_avg
            p_sft_loss = -p_policy_chosen_logps_avg
            n_sft_loss = -n_policy_chosen_logps_avg
            if self.ftx_gamma > 1e-6:
                losses += self.ftx_gamma * (sft_loss+self.dgx_gamma*p_sft_loss+self.dgx_gamma*n_sft_loss)
            metrics["raw_loss"] = losses.mean().cpu()
            if self.have_w:
                # print(f"policy_chosen_logps_avg: {policy_chosen_logps_avg}")
                # print(f"policy_rejected_logps_avg: {policy_rejected_logps_avg}")
                # print(f"losses: {losses}")
                losses = torch.sigmoid(-(self.p_x*policy_chosen_logps_avg+self.n_x*policy_rejected_logps_avg))*losses
            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            p_reward_accuracies = (p_chosen_rewards > p_rejected_rewards).float()
            n_reward_accuracies = (n_chosen_rewards > n_rejected_rewards).float()

            prefix = "eval_" if train_eval == "eval" else ""
            metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
            metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
            metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
            metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).mean().cpu()
            metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
            metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
            metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
            metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()
            
            metrics["{}p_rewards/chosen".format(prefix)] = p_chosen_rewards.mean().cpu()
            metrics["{}p_rewards/rejected".format(prefix)] = p_rejected_rewards.mean().cpu()
            metrics["{}p_rewards/accuracies".format(prefix)] = p_reward_accuracies.mean().cpu()
            metrics["{}p_rewards/margins".format(prefix)] = (p_chosen_rewards - p_rejected_rewards).mean().cpu()
            metrics["{}p_logps/rejected".format(prefix)] = p_policy_rejected_logps.detach().mean().cpu()
            metrics["{}p_logps/chosen".format(prefix)] = p_policy_chosen_logps.detach().mean().cpu()
            metrics["{}p_logits/rejected".format(prefix)] = p_policy_rejected_logits.detach().mean().cpu()
            metrics["{}p_logits/chosen".format(prefix)] = p_policy_chosen_logits.detach().mean().cpu()
            
            metrics["{}n_rewards/chosen".format(prefix)] = n_chosen_rewards.mean().cpu()
            metrics["{}n_rewards/rejected".format(prefix)] = n_rejected_rewards.mean().cpu()
            metrics["{}n_rewards/accuracies".format(prefix)] = n_reward_accuracies.mean().cpu()
            metrics["{}n_rewards/margins".format(prefix)] = (n_chosen_rewards - n_rejected_rewards).mean().cpu()
            metrics["{}n_logps/rejected".format(prefix)] = n_policy_rejected_logps.detach().mean().cpu()
            metrics["{}n_logps/chosen".format(prefix)] = n_policy_chosen_logps.detach().mean().cpu()
            metrics["{}n_logits/rejected".format(prefix)] = n_policy_rejected_logits.detach().mean().cpu()
            metrics["{}n_logits/chosen".format(prefix)] = n_policy_chosen_logits.detach().mean().cpu()
            if self.loss_type == "orpo":
                metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean().cpu()
                metrics["{}odds_ratio_loss".format(prefix)] = ((losses - sft_loss) / self.beta).detach().mean().cpu()

            return losses.mean(), metrics

        else:
            losses, chosen_rewards, rejected_rewards = self.compute_preference_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                reference_chosen_logps,
                reference_rejected_logps,
                p_policy_chosen_logps,
                p_policy_rejected_logps,
                p_reference_chosen_logps,
                p_reference_rejected_logps,
                n_policy_chosen_logps,
                n_policy_rejected_logps,
                n_reference_chosen_logps,
                n_reference_rejected_logps,
            )
            sft_loss = -policy_chosen_logps_avg
            if self.ftx_gamma > 1e-6:
                losses += self.ftx_gamma * (sft_loss)
            if self.have_w:
                losses = torch.sigmoid(-(self.p_x*policy_chosen_logps_avg+self.n_x*policy_rejected_logps_avg))*losses
            reward_accuracies = (chosen_rewards > rejected_rewards).float()

            prefix = "eval_" if train_eval == "eval" else ""
            metrics["{}rewards/chosen".format(prefix)] = chosen_rewards.mean().cpu()
            metrics["{}rewards/rejected".format(prefix)] = rejected_rewards.mean().cpu()
            metrics["{}rewards/accuracies".format(prefix)] = reward_accuracies.mean().cpu()
            metrics["{}rewards/margins".format(prefix)] = (chosen_rewards - rejected_rewards).mean().cpu()
            metrics["{}logps/rejected".format(prefix)] = policy_rejected_logps.detach().mean().cpu()
            metrics["{}logps/chosen".format(prefix)] = policy_chosen_logps.detach().mean().cpu()
            metrics["{}logits/rejected".format(prefix)] = policy_rejected_logits.detach().mean().cpu()
            metrics["{}logits/chosen".format(prefix)] = policy_chosen_logits.detach().mean().cpu()
            
            if self.loss_type == "orpo":
                metrics["{}sft_loss".format(prefix)] = sft_loss.detach().mean().cpu()
                metrics["{}odds_ratio_loss".format(prefix)] = ((losses - sft_loss) / self.beta).detach().mean().cpu()

            return losses.mean(), metrics
