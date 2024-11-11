# Copyright 2024 the LlamaFactory team.
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

from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

from ...extras import logging
from ...extras.constants import IGNORE_INDEX
from .processor_utils import infer_seqlen


if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer, ProcessorMixin

    from ...hparams import DataArguments
    from ..mm_plugin import ImageInput, VideoInput
    from ..template import Template


logger = logging.get_logger(__name__)


def _encode_pairwise_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Tuple[List[int], List[int], List[int], List[int]]:
    chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images, videos, processor)
    rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images, videos, processor)
    prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
    _, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)

    if template.efficient_eos:
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]

    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, videos, tokenizer, processor)
    # consider the response is more important
    source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), cutoff_len)
    prompt_ids = prompt_ids[:source_len]
    chosen_ids = chosen_ids[:target_len]
    rejected_ids = rejected_ids[:target_len]

    chosen_input_ids = prompt_ids + chosen_ids
    chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
    rejected_input_ids = prompt_ids + rejected_ids
    rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels


def preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 2:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels = _encode_pairwise_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
        )
        model_inputs["chosen_input_ids"].append(chosen_input_ids)
        model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
        model_inputs["chosen_labels"].append(chosen_labels)
        model_inputs["rejected_input_ids"].append(rejected_input_ids)
        model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
        model_inputs["rejected_labels"].append(rejected_labels)
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs

def print_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
    valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
    print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
    print("chosen_inputs:\n{}".format(tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False)))
    print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
    print(f"chosen_labels:\n{tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
    print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
    print("rejected_inputs:\n{}".format(tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)))
    print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
    print(f"rejected_labels:\n{tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")

def _sso_encode_pairwise_example(
    prompt: Sequence[Dict[str, str]],
    response: Sequence[Dict[str, str]],
    system: Optional[str],
    p_system: Optional[str],
    n_system: Optional[str],
    tools: Optional[str],
    images: Sequence["ImageInput"],
    videos: Sequence["VideoInput"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    cutoff_len: int,
) -> Tuple[List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[int], List[int]]:
    chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images, videos, processor)
    rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images, videos, processor)
    prompt_ids, chosen_ids = template.encode_oneturn(tokenizer, chosen_messages, system, tools)
    _, rejected_ids = template.encode_oneturn(tokenizer, rejected_messages, system, tools)

    p_chosen_messages = template.mm_plugin.process_messages(prompt + [response[0]], images, videos, processor)
    p_rejected_messages = template.mm_plugin.process_messages(prompt + [response[1]], images, videos, processor)
    p_prompt_ids, p_chosen_ids = template.encode_oneturn(tokenizer, p_chosen_messages, p_system, tools)
    _, p_rejected_ids = template.encode_oneturn(tokenizer, p_rejected_messages, p_system, tools)
    
    n_chosen_messages = template.mm_plugin.process_messages(prompt + [response[2]], images, videos, processor)
    n_rejected_messages = template.mm_plugin.process_messages(prompt + [response[0]], images, videos, processor)
    n_prompt_ids, n_chosen_ids = template.encode_oneturn(tokenizer, n_chosen_messages, n_system, tools)
    _, n_rejected_ids = template.encode_oneturn(tokenizer, n_rejected_messages, n_system, tools)
    
    if template.efficient_eos:
        chosen_ids += [tokenizer.eos_token_id]
        rejected_ids += [tokenizer.eos_token_id]
        p_chosen_ids += [tokenizer.eos_token_id]
        p_rejected_ids += [tokenizer.eos_token_id]
        n_chosen_ids += [tokenizer.eos_token_id]
        n_rejected_ids += [tokenizer.eos_token_id]

    prompt_ids, _ = template.mm_plugin.process_token_ids(prompt_ids, None, images, videos, tokenizer, processor)
    p_prompt_ids, _ = template.mm_plugin.process_token_ids(p_prompt_ids, None, images, videos, tokenizer, processor)
    n_prompt_ids, _ = template.mm_plugin.process_token_ids(n_prompt_ids, None, images, videos, tokenizer, processor)
    # consider the response is more important
    source_len, target_len = infer_seqlen(len(prompt_ids), max(len(chosen_ids), len(rejected_ids)), cutoff_len)
    p_source_len, p_target_len = infer_seqlen(len(p_prompt_ids), max(len(p_chosen_ids), len(p_rejected_ids)), cutoff_len)
    n_source_len, n_target_len = infer_seqlen(len(n_prompt_ids), max(len(n_chosen_ids), len(n_rejected_ids)), cutoff_len)

    prompt_ids = prompt_ids[:source_len]
    chosen_ids = chosen_ids[:target_len]
    rejected_ids = rejected_ids[:target_len]
    
    p_prompt_ids = p_prompt_ids[:p_source_len]
    p_chosen_ids = p_chosen_ids[:p_target_len]
    p_rejected_ids = p_rejected_ids[:p_target_len]
    
    n_prompt_ids = n_prompt_ids[:n_source_len]
    n_chosen_ids = n_chosen_ids[:n_target_len]
    n_rejected_ids = n_rejected_ids[:n_target_len]

    chosen_input_ids = prompt_ids + chosen_ids
    chosen_labels = [IGNORE_INDEX] * source_len + chosen_ids
    rejected_input_ids = prompt_ids + rejected_ids
    rejected_labels = [IGNORE_INDEX] * source_len + rejected_ids
    
    p_chosen_input_ids = p_prompt_ids + p_chosen_ids
    p_chosen_labels = [IGNORE_INDEX] * p_source_len + p_chosen_ids
    p_rejected_input_ids = p_prompt_ids + p_rejected_ids
    p_rejected_labels = [IGNORE_INDEX] * p_source_len + p_rejected_ids
    
    n_chosen_input_ids = n_prompt_ids + n_chosen_ids
    n_chosen_labels = [IGNORE_INDEX] * n_source_len + n_chosen_ids
    n_rejected_input_ids = n_prompt_ids + n_rejected_ids
    n_rejected_labels = [IGNORE_INDEX] * n_source_len + n_rejected_ids
    
    return chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels, p_chosen_input_ids, p_chosen_labels, p_rejected_input_ids, p_rejected_labels, n_chosen_input_ids, n_chosen_labels, n_rejected_input_ids, n_rejected_labels

def sso_preprocess_pairwise_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[Any]]:
    # build input pairs with format `<bos> X`, `Y1 <eos>` and `Y2 <eos>`
    model_inputs = defaultdict(list)
    for i in range(len(examples["_prompt"])):
        if len(examples["_prompt"][i]) % 2 != 1 or len(examples["_response"][i]) < 3:
            logger.warning_rank0(
                "Dropped invalid example: {}".format(examples["_prompt"][i] + examples["_response"][i])
            )
            continue

        chosen_input_ids, chosen_labels, rejected_input_ids, rejected_labels, p_chosen_input_ids, p_chosen_labels, p_rejected_input_ids, p_rejected_labels, n_chosen_input_ids, n_chosen_labels, n_rejected_input_ids, n_rejected_labels = _sso_encode_pairwise_example(
            prompt=examples["_prompt"][i],
            response=examples["_response"][i],
            system=examples["_system"][i],
            p_system=examples["_p_system"][i],
            n_system=examples["_n_system"][i],
            tools=examples["_tools"][i],
            images=examples["_images"][i] or [],
            videos=examples["_videos"][i] or [],
            template=template,
            tokenizer=tokenizer,
            processor=processor,
            cutoff_len=data_args.cutoff_len,
        )
        model_inputs["chosen_input_ids"].append(chosen_input_ids)
        model_inputs["chosen_attention_mask"].append([1] * len(chosen_input_ids))
        model_inputs["chosen_labels"].append(chosen_labels)
        model_inputs["rejected_input_ids"].append(rejected_input_ids)
        model_inputs["rejected_attention_mask"].append([1] * len(rejected_input_ids))
        model_inputs["rejected_labels"].append(rejected_labels)
        
        model_inputs["p_chosen_input_ids"].append(p_chosen_input_ids)
        model_inputs["p_chosen_attention_mask"].append([1] * len(p_chosen_input_ids))
        model_inputs["p_chosen_labels"].append(p_chosen_labels)
        model_inputs["p_rejected_input_ids"].append(p_rejected_input_ids)
        model_inputs["p_rejected_attention_mask"].append([1] * len(p_rejected_input_ids))
        model_inputs["p_rejected_labels"].append(p_rejected_labels)
        
        model_inputs["n_chosen_input_ids"].append(n_chosen_input_ids)
        model_inputs["n_chosen_attention_mask"].append([1] * len(n_chosen_input_ids))
        model_inputs["n_chosen_labels"].append(n_chosen_labels)
        model_inputs["n_rejected_input_ids"].append(n_rejected_input_ids)
        model_inputs["n_rejected_attention_mask"].append([1] * len(n_rejected_input_ids))
        model_inputs["n_rejected_labels"].append(n_rejected_labels)
        
        model_inputs["images"].append(examples["_images"][i])
        model_inputs["videos"].append(examples["_videos"][i])

    return model_inputs

def print_sso_pairwise_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_chosen_labels = list(filter(lambda x: x != IGNORE_INDEX, example["chosen_labels"]))
    valid_rejected_labels = list(filter(lambda x: x != IGNORE_INDEX, example["rejected_labels"]))
    print("chosen_input_ids:\n{}".format(example["chosen_input_ids"]))
    print("chosen_inputs:\n{}".format(tokenizer.decode(example["chosen_input_ids"], skip_special_tokens=False)))
    print("chosen_label_ids:\n{}".format(example["chosen_labels"]))
    print(f"chosen_labels:\n{tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
    print("rejected_input_ids:\n{}".format(example["rejected_input_ids"]))
    print("rejected_inputs:\n{}".format(tokenizer.decode(example["rejected_input_ids"], skip_special_tokens=False)))
    print("rejected_label_ids:\n{}".format(example["rejected_labels"]))
    print(f"rejected_labels:\n{tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
    
    print("p_chosen_input_ids:\n{}".format(example["p_chosen_input_ids"]))
    print("p_chosen_inputs:\n{}".format(tokenizer.decode(example["p_chosen_input_ids"], skip_special_tokens=False)))
    print("p_chosen_label_ids:\n{}".format(example["p_chosen_labels"]))
    print(f"p_chosen_labels:\n{tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
    print("p_rejected_input_ids:\n{}".format(example["p_rejected_input_ids"]))
    print("p_rejected_inputs:\n{}".format(tokenizer.decode(example["p_rejected_input_ids"], skip_special_tokens=False)))
    print("p_rejected_label_ids:\n{}".format(example["p_rejected_labels"]))
    print(f"p_rejected_labels:\n{tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")
    
    print("n_chosen_input_ids:\n{}".format(example["n_chosen_input_ids"]))
    print("n_chosen_inputs:\n{}".format(tokenizer.decode(example["n_chosen_input_ids"], skip_special_tokens=False)))
    print("n_chosen_label_ids:\n{}".format(example["n_chosen_labels"]))
    print(f"n_chosen_labels:\n{tokenizer.decode(valid_chosen_labels, skip_special_tokens=False)}")
    print("n_rejected_input_ids:\n{}".format(example["n_rejected_input_ids"]))
    print("n_rejected_inputs:\n{}".format(tokenizer.decode(example["n_rejected_input_ids"], skip_special_tokens=False)))
    print("n_rejected_label_ids:\n{}".format(example["n_rejected_labels"]))
    print(f"n_rejected_labels:\n{tokenizer.decode(valid_rejected_labels, skip_special_tokens=False)}")