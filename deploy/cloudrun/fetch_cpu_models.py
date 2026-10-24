"""Docker build step for Dockerfile.cpu: the pinned 4-bit base GGUF (sha256-checked), Whisper small
and the tokenizer that sizes chunks, so the container never downloads at start-up."""
import hashlib, shutil
from huggingface_hub import hf_hub_download
from faster_whisper import download_model

gguf = hf_hub_download("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
                       revision="a06e946bb6b655725eafa393f4a9745d460374c9", local_dir="models")
h = hashlib.sha256()
with open(gguf, "rb") as f:
    for block in iter(lambda: f.read(1 << 24), b""):
        h.update(block)
assert h.hexdigest() == "3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597", h.hexdigest()
shutil.rmtree("models/.cache", ignore_errors=True)
download_model("small")
hf_hub_download("Qwen/Qwen3-4B-Instruct-2507", "tokenizer.json")
