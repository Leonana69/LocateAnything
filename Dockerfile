FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel
# This 2.8.0 base is conda-based (/opt/conda) with torch 2.8.0 + cu126 preinstalled.
# The whole image is held at torch 2.8.0 / cu12 so vllm 0.10.x, OmDet, and the
# prebuilt flash-attn 2.8.3 wheel all share one ABI-compatible torch.
# PIP_BREAK_SYSTEM_PACKAGES is a harmless no-op on a conda base (PEP 668 isn't
# enforced); kept so RUN pip installs stay frictionless if the base ever moves
# to a system-Python image.
#
# Verified facts about this base (probed, not assumed):
#   python 3.11.13 (-> cp311 wheels)   numpy 2.3.2
#   torch 2.8.0+cu126   torch._C._GLIBCXX_USE_CXX11_ABI = True (-> cxx11abiTRUE)
# Already present and NOT reinstalled: torch, torchvision, numpy, pillow,
# requests, packaging, setuptools, pip.
ENV PIP_BREAK_SYSTEM_PACKAGES=1 \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DISABLE_REQUIRE=1

# --- Host-driver compatibility (needed when the host driver predates CUDA 12.6) ---
# This box runs driver 535.309.01 (max CUDA 12.2) on a GeForce RTX 4090, but the
# base image is CUDA 12.6. Two things otherwise block `docker run --gpus all`:
#   1. The container toolkit refuses the image because its baked
#      NVIDIA_REQUIRE_CUDA says cuda>=12.6  ->  NVIDIA_DISABLE_REQUIRE=1 skips that
#      gate (set above). torch's bundled cu126 runtime then uses CUDA *Minor
#      Version Compatibility* to run on the 12.2 driver (same as the host conda env).
#   2. The base ships forward-compat driver libs (/usr/local/cuda/compat,
#      libcuda.so.560). On an older driver the toolkit injects those as the active
#      libcuda, but *Forward* Compatibility is datacenter-GPU only -> on a GeForce
#      it dies with "CUDA error 804: forward compatibility was attempted on non
#      supported HW". Removing them forces use of the host's libcuda.535 (MVC),
#      which is GeForce-safe. Harmless no-op on a host whose driver is >= 12.6.
RUN rm -rf /usr/local/cuda/compat /usr/local/cuda-12.6/compat

# No apt packages are needed: opencv-python-headless ships no libGL dependency and
# decord / lmdb / flash-attn arrive as self-contained manylinux wheels.

# Minimal runtime deps for scripts/locateanything_server.py. These are the only
# packages the server's import chain actually pulls in beyond the base image:
#   transformers/tokenizers/peft  - model load + the vendored LoRA-wrapped towers
#   opencv-python-headless/decord/lmdb - imported at module load by the processor
#   flask/werkzeug                - the HTTP server itself
#   einops                        - required by flash_attn.bert_padding
#   bitsandbytes                  - optional int8 / 4-bit weight quantization
#                                   (--load-in-8bit / --load-in-4bit; ~halves GPU mem)
# Pins mirror pyproject.toml. scipy and gunicorn are declared there but never
# imported by the server path, so they are intentionally omitted to stay minimal.
RUN pip install --no-cache-dir \
        transformers==4.57.1 \
        tokenizers==0.22.0 \
        peft==0.12.0 \
        opencv-python-headless \
        decord \
        lmdb \
        flask \
        werkzeug \
        einops \
        bitsandbytes==0.49.2

# FlashAttention-2 2.8.3 — prebuilt wheel matching the base exactly:
# cu12 / torch2.8 / cxx11abiTRUE / cp311. --no-deps so it cannot drag torch back
# in. Only the Moon-ViT vision tower uses FA2 (~3-4x faster prefill, <1/2 the GPU
# memory); the Qwen2 decoder's Parallel Box Decoding stays on sdpa — LocateAnything
# Worker pins this per tower, so installing FA2 never routes the decoder to it.
RUN pip install --no-cache-dir --no-deps \
        "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"

# Copy the source last so editing it doesn't invalidate the dependency layers.
WORKDIR /opt/locateanything
COPY . /opt/locateanything
# Install the package itself without deps (handled above) so `import locateanything`
# resolves regardless of CWD / launcher (python or gunicorn).
RUN pip install --no-cache-dir --no-deps .

# Server defaults (override at `docker run` time). run-server.sh reads these.
ENV MODEL_PATH=nvidia/LocateAnything-3B \
    HOST=0.0.0.0 \
    PORT=8000 \
    DEVICE=cuda \
    DTYPE=bfloat16
EXPOSE 8000

# Needs a GPU at runtime: `docker run --gpus all -p 8000:8000 <image>`
CMD ["bash", "scripts/run-server.sh"]
