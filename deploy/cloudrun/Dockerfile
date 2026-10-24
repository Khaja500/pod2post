# The website (app/gradio_app.py) on a Cloud Run L4 GPU: vLLM with both trained adapters,
# faster-whisper on the same GPU. Built by deploy/cloudrun/deploy.sh, which stages the build
# context (app/, src/, configs/, adapters/) so the large local data/ and runs/ folders stay out.
#
# CUDA 12.9 build of the vLLM image used on Kaggle: ctranslate2 (faster-whisper) needs CUDA 12 + cuDNN 9,
# which torch's pip wheels in this image provide.
FROM vllm/vllm-openai:v0.32.0-x86_64-cu129

ENV HF_HOME=/models/hf \
    PYTHONUNBUFFERED=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=8080

RUN PKGS="gradio==6.30.0 faster-whisper==1.2.1 av==18.1.0 yt-dlp==2026.10.21 textstat==0.7.13 spacy==3.8.16 \
      https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl" \
 && if command -v uv >/dev/null; then uv pip install --system --no-cache $PKGS; else python3 -m pip install --no-cache-dir $PKGS; fi \
 && python3 -c "import vllm, gradio, faster_whisper, spacy; spacy.load('en_core_web_sm'); print('vllm', vllm.__version__, 'gradio', gradio.__version__)"

# Weights inside the image: Cloud Run's writable disk is RAM, so downloading at start-up would cost memory and time.
RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-4B-Instruct-2507'); \
from faster_whisper import download_model; download_model('large-v3-turbo')"
ENV HF_HUB_OFFLINE=1

WORKDIR /app
COPY adapters/ /app/adapters/
COPY configs/ /app/configs/
COPY src/ /app/src/
COPY app/ /app/app/

ENTRYPOINT []
# --daily-limit caps GPU time on the public site; --max-minutes refuses very long episodes.
CMD ["sh", "-c", "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv; exec python3 app/gradio_app.py --backend vllm --adapter /app/adapters/sft-dpo --adapter /app/adapters/sft --daily-limit ${DAILY_LIMIT:-25} --max-minutes ${MAX_MINUTES:-120}"]
