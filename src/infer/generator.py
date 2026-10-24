"""vLLM chat generation shared by the teacher (02), evaluation (05) and the demo.

Prompts are rendered with the model's own chat template (thinking disabled for
Qwen3), so every caller sends exactly the same text for the same messages.
LlamaCppGenerator runs the same model on machines without an NVIDIA GPU (the website on a
Mac): a 4-bit GGUF of the base model in llama.cpp's llama-server, with the LoRA adapters
converted to GGUF and picked per request.
"""
from __future__ import annotations

import tempfile
import time
import zlib
from pathlib import Path


class VLLMGenerator:
    def __init__(self, model: str, tensor_parallel_size: int = 1, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.9, seed: int = 42, max_lora_rank: int = 0,
                 engine: dict | None = None, **sampling):
        """engine: extra vllm.LLM arguments (e.g. CUDA-graph sizes for a faster start-up)."""
        from vllm import LLM

        kwargs = dict(enable_lora=True, max_lora_rank=max_lora_rank, max_loras=1) if max_lora_rank else {}
        kwargs.update(engine or {})
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


class LlamaCppGenerator:
    """Same interface as VLLMGenerator, backed by a llama-server process it starts and stops.

    Adapters are loaded at start with scale 0 and switched on per request, so one server
    serves the base model and every adapter. Requests run one at a time (`--parallel 1`):
    an 8 GB Mac has room for one 8k-token context next to the 2.5 GB model."""

    def __init__(self, model_gguf: str, lora_ggufs: list[str] = (), count=None, port: int = 8089,
                 ctx: int = 8192, timeout_s: float = 3600, binary: str = "llama-server", **sampling):
        import atexit
        import subprocess

        import requests

        self.url, self.http, self.timeout_s = f"http://127.0.0.1:{port}", requests.Session(), timeout_s
        cmd = [binary, "-m", model_gguf, "--host", "127.0.0.1", "--port", str(port), "-c", str(ctx),
               "--parallel", "1", "-ngl", "99", "--jinja", "--no-webui"]
        for lora in lora_ggufs:
            cmd += ["--lora", lora]
        if lora_ggufs:
            cmd.append("--lora-init-without-apply")
        self.log = Path(tempfile.gettempdir()) / f"llama-server-{port}.log"  # a pipe nobody reads would fill up
        with self.log.open("w") as log:
            self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        atexit.register(self.close)
        self._wait_ready()
        self.adapter_ids = {str(Path(a["path"]).resolve()): a["id"] for a in
                            (self.http.get(f"{self.url}/lora-adapters", timeout=10).json() if lora_ggufs else [])}
        self.count = count or self._count
        self.sampling = {"temperature": 0.7, "top_p": 0.8, "top_k": 20, **sampling}
        self.generated_tokens, self.generate_seconds, self.lora = 0, 0.0, None

    def _wait_ready(self, limit_s: float = 300) -> None:
        import requests

        t0 = time.time()
        while time.time() - t0 < limit_s:
            if self.proc.poll() is not None:
                raise RuntimeError(f"llama-server exited; see {self.log}:\n" + self.log.read_text()[-2000:])
            try:
                if self.http.get(f"{self.url}/health", timeout=2).status_code == 200:
                    return
            except requests.ConnectionError:
                pass
            time.sleep(1)
        raise RuntimeError(f"llama-server did not become ready in {limit_s:.0f}s")

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=30)

    def use_adapter(self, name: str | None, path: str | None = None) -> None:
        """Switch to a loaded GGUF adapter by its path (None = base model)."""
        self.lora = self.adapter_ids[str(Path(path).resolve())] if name else None

    def _count(self, text: str) -> int:
        r = self.http.post(f"{self.url}/tokenize", json={"content": text, "add_special": False}, timeout=60)
        return len(r.json()["tokens"])

    def chat(self, conversations: list[list[dict]], max_tokens: int, temperature: float | None = None) -> list[str]:
        sampling = {**self.sampling, **({"temperature": temperature} if temperature is not None else {})}
        lora = [{"id": i, "scale": 1.0 if i == self.lora else 0.0} for i in sorted(self.adapter_ids.values())]
        outs = []
        for conv in conversations:
            t0 = time.time()
            r = self.http.post(f"{self.url}/v1/chat/completions", timeout=self.timeout_s,
                               json={"messages": conv, "max_tokens": max_tokens, "lora": lora, **sampling})
            r.raise_for_status()
            body = r.json()
            self.generate_seconds += time.time() - t0
            self.generated_tokens += body.get("usage", {}).get("completion_tokens", 0)
            outs.append(body["choices"][0]["message"]["content"] or "")
        return outs
