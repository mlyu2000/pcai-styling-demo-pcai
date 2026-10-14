# PCAI Styling Demo — vLLM-Omni I2V inference image.
# Model weights are NOT baked in: they download at pod start from HuggingFace
# into the PVC-mounted /models (see manifests/inference.yaml).
#
# Build modes (set via build args):
#   BASE_IMAGE=vllm/vllm-omni:latest            (preferred, if published)
#   BASE_IMAGE=vllm/vllm-openai:latest          (fallback: install vllm-omni on top)
ARG BASE_IMAGE=vllm/vllm-openai:latest
FROM ${BASE_IMAGE}

ARG INSTALL_OMNI=true
# Ensure the omni (diffusion/video) stack is present; no-op if the base already has it.
RUN if [ "$INSTALL_OMNI" = "true" ]; then \
      python -c "import vllm_omni" 2>/dev/null || pip install --no-cache-dir vllm-omni; \
    fi

WORKDIR /app
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

ENV MODEL_DIR=/models \
    HF_HOME=/models/hf \
    VLLM_USE_V1=1 \
    TOKENIZERS_PARALLELISM=false

ENTRYPOINT ["/app/entrypoint.sh"]
