from typing import List
import os
import torch
from transformers import AutoTokenizer
from typing import List
import math
import ray
from vllm import LLM, SamplingParams
import socket


class MultiProcessVllmInferencer:
    def __init__(
        self,
        model_path: str,
        num_gpus_per_model: int = 1,
        do_sample: bool = False,
        num_beams: int = 1,
        max_new_tokens: int = 1024,
        temperature: float = 0,
        top_p: float = 1.0,
        top_k: int = -1,
        frequency_penalty=0.0,
        stop: List[str] = [],
    ):

        self.num_gpus_total = torch.cuda.device_count()
        self.num_gpus_per_model = num_gpus_per_model

        self.model_path = model_path

        self.sampling_params = SamplingParams(
            max_tokens=max_new_tokens,
            use_beam_search=(not do_sample) and (not num_beams == 1),
            best_of=num_beams,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            length_penalty=1.0,
            frequency_penalty=frequency_penalty,
            stop=stop,
            early_stopping=False,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        ray.init(ignore_reinit_error=True)

    def find_n_free_ports(self, n):
        ports = []
        sockets = []
        port_range_start = 5000
        port_range_end = 8000
        current_port = port_range_start

        while len(ports) < n and current_port <= port_range_end:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(("", current_port))
                ports.append(current_port)
                sockets.append(s)
            except OSError:
                # 如果端口已经被占用，继续尝试下一个端口
                pass
            current_port += 1

        if len(ports) < n:
            raise RuntimeError(
                f"Could only find {len(ports)} free ports within the specified range."
            )

        return ports, sockets

    @staticmethod
    def single_process_inference(
        model_path, num_gpus_per_model, vllm_port, *args, **kwargs
    ):

        # Set an available port
        os.environ["VLLM_PORT"] = str(vllm_port)
        print(f"Using VLLM_PORT: {vllm_port}")

        model = LLM(
            model=model_path,
            tensor_parallel_size=num_gpus_per_model,
            enforce_eager=True,
            trust_remote_code=True
        )

        return model.generate(*args, **kwargs)

    def inference(self, data: List[str], sampling_params:SamplingParams=None, use_tqdm=True):
        get_answers_func = ray.remote(num_gpus=self.num_gpus_per_model)(
                MultiProcessVllmInferencer.single_process_inference
            ).remote

        num_processes = min(
            len(data), max(1, self.num_gpus_total // self.num_gpus_per_model)
        )
        chunk_size = math.ceil(len(data) / num_processes)

        ports, sockets = self.find_n_free_ports(num_processes)

        if sampling_params is None:
            sampling_params = self.sampling_params
        
        gathered_responses = []
        for idx, i in enumerate(range(0, len(data), chunk_size)):
            gathered_responses.append(
                get_answers_func(
                    self.model_path,
                    self.num_gpus_per_model,
                    ports[idx],
                    data[i : i + chunk_size],
                    sampling_params,
                    use_tqdm=use_tqdm,
                )
            )

        for s in sockets:
            s.close()
            
        gathered_responses = ray.get(gathered_responses)

        gathered_responses = [
            item for sublist in gathered_responses for item in sublist
        ]
        return gathered_responses

    def get_tokenizer(self):
        return self.tokenizer
