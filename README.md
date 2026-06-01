# LocateAnything (inference library)

A trimmed, self-contained packaging of [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B)
for use as a Python library inside a vision service. Only the inference code is
included — training, evaluation, docs, and assets have been removed.

LocateAnything is a vision-language grounding model (Moon-ViT encoder + Qwen
decoder) that does object detection, phrase grounding, GUI grounding, text
detection, and pointing, using **Parallel Box Decoding** for fast box output.

## What's inside

```
locateanything/
├── __init__.py            # exports LocateAnythingWorker
├── worker.py              # the inference API
└── model/                 # vendored model code (Moon-ViT + Qwen + PBD)
    ├── __init__.py        # registers classes with transformers Auto* registries
    ├── modeling_*.py, configuration_*.py, processing_*.py, generate_utils.py, ...
    └── *.json             # chat template + processor / preprocessor configs
```

The model code is vendored locally and registered with the `transformers`
`Auto*` factories, so weights load **without `trust_remote_code=True`** — no code
is downloaded from the Hub at runtime.

## Install

```bash
pip install -e .
# optional speedups (Hopper/Blackwell): pip install -e ".[speed]"
```

`flash-attn` / `magi-attention` are optional; the model falls back to SDPA
attention when they are not installed.

## Usage

```python
from PIL import Image
from locateanything import LocateAnythingWorker

worker = LocateAnythingWorker("nvidia/LocateAnything-3B")  # or a local weights dir
img = Image.open("example.jpg").convert("RGB")

# Object detection
out = worker.detect(img, ["person", "car", "bicycle"])
print(out["answer"])

# Phrase grounding (multiple instances)
out = worker.ground_multi(img, "people wearing red shirts")

# GUI grounding as a point
out = worker.ground_gui(img, "the search button", output_type="point")

# Parse structured output into pixel coordinates
w, h = img.size
boxes = LocateAnythingWorker.parse_boxes(out["answer"], w, h)
points = LocateAnythingWorker.parse_points(out["answer"], w, h)
```

`generation_mode` can be `"fast"` (MTP — all box coords in one parallel pass),
`"slow"` (autoregressive), or `"hybrid"` (default; MTP with autoregressive
fallback on malformed boxes).

## Test server

A small Flask server lives under `scripts/` for quick manual testing — load the
model once, then upload an image in the browser or POST to the JSON endpoints.

```bash
pip install -e ".[serve]"           # adds flask + gunicorn
bash scripts/run-server.sh          # MODEL_PATH=... PORT=8080 to override
# or: python scripts/locateanything_server.py --model-path nvidia/LocateAnything-3B
```

Then open <http://localhost:8000/> for the web UI, or call the API:

```bash
curl -F image=@example.jpg -F categories="person,car" http://localhost:8000/detect
```

Endpoints: `/predict`, `/detect`, `/ground_single`, `/ground_multi`,
`/ground_text`, `/detect_text`, `/ground_gui`, `/point`, plus `/health` and a
`/api` listing. The server runs `python scripts/locateanything_server.py`
directly (it adds the repo root to `sys.path`), so an editable install isn't
strictly required for the library — only `[serve]` for Flask itself.

## Notes

- The first call downloads the weights from the Hub (or point `model_path` at a
  local directory) — only the *code* is vendored, not the weights.
- Requires a CUDA GPU and `bfloat16` support for the default configuration.

## License

Released under the [MIT License](./LICENSE). The vendored model code originates
from [NVIDIA Eagle / LocateAnything](https://github.com/NVlabs/EAGLE) (Apache-2.0).
