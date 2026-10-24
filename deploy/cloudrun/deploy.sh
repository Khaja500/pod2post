#!/usr/bin/env bash
# Build the website image with Cloud Build and deploy it to Cloud Run.
#
#   PROJECT=pod2post-c24902 deploy/cloudrun/deploy.sh                 # one L4 GPU: vLLM + both adapters
#   PROJECT=pod2post-c24902 MODE=cpu deploy/cloudrun/deploy.sh        # no GPU: llama.cpp + the GGUF adapter
#   PROJECT=pod2post-c24902 MODE=kaggle KAGGLE_USERNAME=<you> deploy/cloudrun/deploy.sh
#                                    # front end only: every article runs on your Kaggle GPUs (07_infer)
#   PROJECT=pod2post-c24902 SKIP_BUILD=1 deploy/cloudrun/deploy.sh    # redeploy the last image
#   PROJECT=pod2post-c24902 BUILD_ONLY=1 deploy/cloudrun/deploy.sh    # image only (no GPU quota needed)
#
# Cost control: at most 1 instance, scale to zero when idle (billed only while an instance is up,
# including ~15 min idle before it stops), and the app's --daily-limit (DAILY_LIMIT).
# MODE=cpu needs the adapter GGUF in models/ (README, "Run it on a Mac"). MODE=kaggle needs the adapters
# on Kaggle (`scripts/kaggle_infer.py adapters`) and a Kaggle API token in Secret Manager as
# `kaggle-api-token`, which the service reads as KAGGLE_API_TOKEN (README, "Public site on Google Cloud Run").
set -euo pipefail

PROJECT=${PROJECT:?set PROJECT}
MODE=${MODE:-gpu}
REGION=${REGION:-us-central1}
SERVICE=${SERVICE:-pod2post}
DPO_ADAPTER=${DPO_ADAPTER:-runs/2026-10-23/04_dpo-v4/adapter}
SFT_ADAPTER=${SFT_ADAPTER:-runs/2026-10-22/03_sft-v2/adapter}
IMAGE="$REGION-docker.pkg.dev/$PROJECT/pod2post/site-$MODE:latest"
REPO=$(cd "$(dirname "$0")/../.." && pwd)

if [ -z "${SKIP_BUILD:-}" ]; then
  gcloud artifacts repositories describe pod2post --location "$REGION" --project "$PROJECT" >/dev/null 2>&1 \
    || gcloud artifacts repositories create pod2post --repository-format docker --location "$REGION" --project "$PROJECT"
  ctx=$(mktemp -d)
  trap 'rm -rf "$ctx"' EXIT
  cp -R "$REPO/app" "$REPO/src" "$REPO/configs" "$ctx/"
  if [ "$MODE" = kaggle ]; then
    cp "$REPO/deploy/cloudrun/Dockerfile.kaggle" "$ctx/Dockerfile"
    cp "$REPO/requirements-kaggle.txt" "$ctx/"
    mkdir -p "$ctx/kaggle" "$ctx/scripts"
    cp -R "$REPO/kaggle/07_infer" "$REPO/kaggle/07_deps" "$ctx/kaggle/"
    cp "$REPO"/scripts/{build_notebook,kaggle_run,kaggle_infer}.py "$ctx/scripts/"
  elif [ "$MODE" = cpu ]; then
    cp "$REPO/deploy/cloudrun/Dockerfile.cpu" "$ctx/Dockerfile"
    cp "$REPO/deploy/cloudrun/fetch_cpu_models.py" "$ctx/"
    mkdir "$ctx/models"
    cp "$REPO"/models/*-lora-*.gguf "$ctx/models/"
  else
    cp "$REPO/deploy/cloudrun/Dockerfile" "$ctx/"
    for pair in "sft-dpo:$DPO_ADAPTER" "sft:$SFT_ADAPTER"; do
      mkdir -p "$ctx/adapters/${pair%%:*}"
      cp "$REPO/${pair#*:}/adapter_config.json" "$REPO/${pair#*:}/adapter_model.safetensors" "$ctx/adapters/${pair%%:*}/"
    done
  fi
  find "$ctx" -name __pycache__ -prune -exec rm -rf {} +
  gcloud builds submit "$ctx" --project "$PROJECT" --region "$REGION" --tag "$IMAGE" \
    --machine-type e2-highcpu-8 --disk-size 200 --timeout 2h
fi
[ -n "${BUILD_ONLY:-}" ] && exit 0

secrets=()
if [ "$MODE" = kaggle ]; then
  # Waits on Kaggle. Uploads (<= MAX_FILE_MB) sit in memory, Cloud Run's disk, while they go to Kaggle.
  # HTTP/2 end to end lifts Cloud Run's 32 MB HTTP/1 request limit; the app serves it with Hypercorn.
  hardware=(--cpu 2 --memory 8Gi --use-http2)
  env="DAILY_LIMIT=${DAILY_LIMIT:-10},MAX_FILE_MB=${MAX_FILE_MB:-500},KAGGLE_USERNAME=${KAGGLE_USERNAME:?set KAGGLE_USERNAME}"
  secrets=(--set-secrets KAGGLE_API_TOKEN=kaggle-api-token:latest)
elif [ "$MODE" = cpu ]; then
  # 8 vCPUs is Cloud Run's most without a GPU. Threads are pinned because a container sees the host's
  # cores; 4 (one per physical core) beat 8 by ~20% on the same transcript (443 s vs 552 s).
  hardware=(--cpu 8 --memory 16Gi --no-use-http2)
  env="DAILY_LIMIT=${DAILY_LIMIT:-20},MAX_MINUTES=${MAX_MINUTES:-30},LLAMA_ARG_THREADS=4,OMP_NUM_THREADS=4"
else
  hardware=(--gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy --cpu 8 --memory 32Gi --no-use-http2)
  env="DAILY_LIMIT=${DAILY_LIMIT:-25},MAX_MINUTES=${MAX_MINUTES:-120}"
fi
gcloud run deploy "$SERVICE" --project "$PROJECT" --region "$REGION" --image "$IMAGE" "${hardware[@]}" \
  --no-cpu-throttling --execution-environment gen2 \
  --min-instances 0 --max-instances 1 --concurrency 20 --timeout 3600 --port 8080 \
  --startup-probe tcpSocket.port=8080,periodSeconds=10,timeoutSeconds=5,failureThreshold=60 \
  --set-env-vars "$env" ${secrets[@]+"${secrets[@]}"} --allow-unauthenticated
gcloud run services describe "$SERVICE" --project "$PROJECT" --region "$REGION" --format 'value(status.url)'
