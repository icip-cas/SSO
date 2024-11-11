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

from typing import TYPE_CHECKING, Dict, Literal, Optional, Tuple, Union

import torch

from ...extras.constants import IGNORE_INDEX
from ..trainer_utils import get_batch_logps

from .trainer import CustomDPOTrainer

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from ...hparams import FinetuningArguments

class MyCustomDPOTrainer(CustomDPOTrainer):
    def __init__(
        self,
        model: Union["PreTrainedModel", torch.nn.Module],
        ref_model: Optional[Union["PreTrainedModel", torch.nn.Module]],
        finetuning_args: "FinetuningArguments",
        processor: Optional["ProcessorMixin"],
        disable_dropout: bool = True,
        **kwargs,
    ):
        super().__init__(model, ref_model, finetuning_args, processor, disable_dropout, **kwargs)


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
            if self.finetuning_args.G_function:
                p_chosen_rewards = self.beta * p_policy_chosen_logps.to(self.accelerator.device).detach()
                p_rejected_rewards = self.beta * p_policy_rejected_logps.to(self.accelerator.device).detach()
                n_chosen_rewards = self.beta * n_policy_chosen_logps.to(self.accelerator.device).detach()
                n_rejected_rewards = self.beta * n_policy_rejected_logps.to(self.accelerator.device).detach()
                exlosses = self.finetuning_args.G_beta*p_losses+self.finetuning_args.G_beta*n_losses
                return losses,exlosses, chosen_rewards, rejected_rewards, p_chosen_rewards, p_rejected_rewards, n_chosen_rewards, n_rejected_rewards
            else:
                return losses,None, chosen_rewards, rejected_rewards
        else:
            losses, chosen_rewards, rejected_rewards= self.dpo_loss(
                policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps
            )
            if self.finetuning_args.G_function:
                p_losses, p_chosen_rewards, p_rejected_rewards= self.dpo_loss(
                    p_policy_chosen_logps, p_policy_rejected_logps, p_reference_chosen_logps, p_reference_rejected_logps
                )
                n_losses, n_chosen_rewards, n_rejected_rewards= self.dpo_loss(
                    n_policy_chosen_logps, n_policy_rejected_logps, n_reference_chosen_logps, n_reference_rejected_logps
                ) 
                exlosses = self.finetuning_args.G_beta*p_losses+self.finetuning_args.G_beta*n_losses
                return losses, exlosses, chosen_rewards, rejected_rewards, p_chosen_rewards, p_rejected_rewards, n_chosen_rewards, n_rejected_rewards
            else:
                return losses, None, chosen_rewards, rejected_rewards

    def concatenated_forward(
        self, model: "PreTrainedModel", batch: Dict[str, "torch.Tensor"]
    ):
        r"""
        Computes the sum log probabilities of the labels under given logits if loss_type is not IPO, ORPO or SimPO.

        Otherwise the average log probabilities.
        """
        if self.finetuning_args.use_ref_model:
            batch = {k: v.detach().clone() for k, v in batch.items()}  # avoid error

        batch_size = batch["input_ids"].size(0)
        third_size = batch_size // 3

        if self.finetuning_args.G_function:
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
        if self.finetuning_args.G_function:
            losses, exlosses, chosen_rewards, rejected_rewards, p_chosen_rewards, p_rejected_rewards, n_chosen_rewards, n_rejected_rewards = self.compute_preference_loss(
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
                losses += self.ftx_gamma * (sft_loss+self.finetuning_args.G_beta*p_sft_loss+self.finetuning_args.G_beta*n_sft_loss)
            metrics["raw_loss"] = losses.mean().cpu()
            if self.finetuning_args.W_function:
                losses = torch.sigmoid(-(self.finetuning_args.W_alpha*policy_chosen_logps_avg+(1-self.finetuning_args.W_alpha)*policy_rejected_logps_avg))*exlosses+losses
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
            losses, _, chosen_rewards, rejected_rewards = self.compute_preference_loss(
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
            if self.finetuning_args.W_function:
                losses = torch.sigmoid(-(self.finetuning_args.W_alpha*policy_chosen_logps_avg+(1-self.finetuning_args.W_alpha)*policy_rejected_logps_avg))*losses
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
