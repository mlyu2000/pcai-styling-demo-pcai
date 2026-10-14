#!/usr/bin/env bash
# PCAI Styling Demo — inference container entrypoint (vLLM-Omni).
# Serves the selected I2V model through the OpenAI-compatible /v1/videos API.
# Model weights are downloaded from HuggingFace into the PVC-mounted /models
# (so they persist across pod restarts).
#
# Tunables (env): MODEL_NAME, FP8, FLOW_SHIFT, GPU_MEMORY_UTILIZATION, PORT,
#                 TENSOR_PARALLEL, MAX_CONCURRENT_GENERATIONS, MODEL_DIR
set -euo pipefail

: "${MODEL_NAME:=hunyuanvideo-community/HunyuanVideo-1.5-Diffusers-480p_i2v}"
: "${MODEL_DIR:=/models}"
: "${PORT:=8000}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"
: "${MAX_CONCURRENT_GENERATIONS:=1}"
: "${FP8:=1}"            # 1 = --quantization fp8 (use for Hunyuan on A100-40G)
: "${FLOW_SHIFT:=}"      # Hunyuan 480p i2v = 5.0 ; Wan2.2 i2v = 12.0
: "${TENSOR_PARALLEL:=1}"

mkdir -p "$MODEL_DIR"

echo "[entrypoint] MODEL_NAME=$MODEL_NAME GPU_MEM_UTIL=$GPU_MEMORY_UTILIZATION FP8=$FP8 TP=$TENSOR_PARALLEL FLOW_SHIFT=${FLOW_SHIFT:-n/a}"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

# Pre-download the model so the engine starts fast and uses the local copy.
python - <<'PY' || echo "[entrypoint] model pre-download failed (will rely on HF cache at serve time)"
import os
from huggingface_hub import snapshot_download
name = os.environ.get("MODEL_NAME")
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None
model_dir = os.path.join(os.environ.get("MODEL_DIR", "/models"), "weights")
p = snapshot_download(name, local_dir=model_dir, token=token, resume_download=True)
print("[entrypoint] model cached at", p)
PY

MODEL_PATH="$MODEL_DIR/weights"
[ -d "$MODEL_PATH" ] || MODEL_PATH="$MODEL_NAME"

ARGS=(
  serve "$MODEL_PATH"
  --omni
  --host 0.0.0.0
  --port "$PORT"
  --tensor-parallel-size "$TENSOR_PARALLEL"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --max-num-seqs "$MAX_CONCURRENT_GENERATIONS"
  --trust-remote-code
)

[ -n "$FP8" ] && [ "$FP8" = "1" ] && ARGS+=(--quantization fp8)
[ -n "${FLOW_SHIFT:-}" ] && ARGS+=(--flow-shift "$FLOW_SHIFT")

echo "[entrypoint] vllm serve: ${ARGS[*]}"
exec vllm "${ARGS[@]}"
