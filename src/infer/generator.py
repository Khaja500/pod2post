"""vLLM chat generation shared by the teacher (02), evaluation (05) and the demo.

Prompts are rendered with the model's own chat template (thinking disabled for
Qwen3), so every caller sends exactly the same text for the same messages.
"""
from __future__ import annotations

import time
import zlib


class VLLMGenerator:
    def __init__(self, model: str, tensor_parallel_size: int = 1, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.9, seed: int = 42, max_lora_rank: int = 0, **sampling):
        from vllm import LLM

        kwargs = dict(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1) if max_lora_rank else {}
        self.llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size, dtype="float16",
                       max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
                       seed=seed, **kwargs)
        self.tok = self.llm.get_tokenizer()
        self.sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, **sampling}
        self.generated_tokens = 0
        self.generate_seconds = 0.0
        self.lora = None

    def use_adapter(self, name: str | None, path: str | None = None) -> None:
        """Switch to a LoRA adapter (None = base model)."""
        from vllm.lora.request import LoRARequest

        self.lora = LoRARequest(name, zlib.crc32(name.encode()) % 100_000 + 1, path) if name else None

    def count(self, text: str) -> int:
        return len(self.tok.encode(text, add_special_tokens=False))

    def chat(self, conversations: list[list[dict]], max_tokens: int, temperature: float | None = None) -> list[str]:
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=max_tokens, **{**self.sampling,
                                **({"temperature": temperature} if temperature is not None else {})})
        prompts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True,
                                                enable_thinking=False) for c in conversations]
        t0 = time.time()
        outs = self.llm.generate(prompts, params, use_tqdm=False, lora_request=self.lora)
        self.generate_seconds += time.time() - t0
        self.generated_tokens += sum(len(o.outputs[0].token_ids) for o in outs)
        return [o.outputs[0].text for o in outs]
