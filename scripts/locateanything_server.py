# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
locateanything_server.py - A simple Flask server exposing LocateAnything inference.

The model is loaded once at startup and reused for every request. GPU access is
serialized with a lock so the single model instance is shared safely across the
threaded dev server.

Run:
    python locateanything_server.py --model-path nvidia/LocateAnything-3B \
        --host 0.0.0.0 --port 8000

Endpoints:
    GET  /                 web UI — upload an image and draw the result
    GET  /api              machine-readable endpoint listing (JSON)
    GET  /health           liveness / model-loaded status
    POST /predict          generic: image + free-form `question`
    POST /detect           image + `categories` (list or comma-separated string)
    POST /ground_single    image + `phrase`         (single instance)
    POST /ground_multi     image + `phrase`         (all instances)
    POST /ground_text      image + `phrase`         (text grounding)
    POST /detect_text      image                    (scene text detection)
    POST /ground_gui       image + `phrase` [+ `output_type`=box|point]
    POST /point            image + `phrase`         (pointing)

Image input — provide exactly one of (checked in this order):
    1. multipart/form-data file field "image" (or "file")
    2. base64 string in field "image" (a leading `data:` URI prefix is stripped)
    3. "image_url": an http(s) URL (needs --allow-url-fetch) or a local filesystem
       path (needs --allow-local-paths). Both are disabled by default.

Generation params (optional, accepted on every POST):
    generation_mode = fast | slow | hybrid   (default: hybrid)
    max_new_tokens  = int                     (default: worker default, 2048)
    temperature     = float                   (default: worker default, 0.7)
    verbose         = bool                     (default: true -> include timing stats)

Serving under gunicorn (single worker — the model is large and not fork-safe):
    pip install -e ".[serve]"   # provides gunicorn
    LA_MODEL_PATH=nvidia/LocateAnything-3B gunicorn -w 1 -t 0 -b 0.0.0.0:8000 \
        locateanything_server:app
    (any --flag below has an LA_<FLAG> env equivalent, e.g. LA_DEVICE, LA_DTYPE)
"""
import argparse
import base64
import binascii
import io
import os
import threading
import time
from typing import Callable, Optional

import sys

import torch
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image, UnidentifiedImageError
from werkzeug.exceptions import BadRequest, HTTPException, ServiceUnavailable

# Allow running directly (``python scripts/locateanything_server.py``) without an
# editable install by putting the repo root on the path; harmless if installed.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from locateanything import LocateAnythingWorker

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
VALID_MODES = {"fast", "slow", "hybrid"}

DEFAULT_MAX_IMAGE_MB = float(os.environ.get("LA_MAX_IMAGE_MB", "25"))
DEFAULT_MAX_IMAGE_PIXELS = int(os.environ.get("LA_MAX_IMAGE_PIXELS", str(50_000_000)))
DEFAULT_MAX_FETCH_SECONDS = float(os.environ.get("LA_MAX_FETCH_SECONDS", "30"))


def _body_cap(max_image_mb: float) -> int:
    # base64 inflates ~33%; leave headroom for multipart overhead + text fields.
    return int(max_image_mb * 1024 * 1024 * 1.4) + 1024 * 1024


app = Flask(__name__)
# Cap request bodies even before the model loads (e.g. a WSGI import without
# LA_MODEL_PATH set); load_model refreshes this from --max-image-mb.
app.config["MAX_CONTENT_LENGTH"] = _body_cap(DEFAULT_MAX_IMAGE_MB)

# ---- Global, process-wide state (one shared model) ----
_worker: Optional[LocateAnythingWorker] = None
_infer_lock = threading.Lock()  # serialize GPU access across request threads
_config = {
    "model_path": None,
    "device": None,
    "dtype": None,
    "quantization": None,
    "allow_url_fetch": False,
    "allow_local_paths": False,
    "max_image_bytes": int(DEFAULT_MAX_IMAGE_MB * 1024 * 1024),
    "max_image_pixels": DEFAULT_MAX_IMAGE_PIXELS,
    "max_fetch_seconds": DEFAULT_MAX_FETCH_SECONDS,
}


def load_model(model_path: str, device: str = "cuda", dtype: str = "bfloat16",
               allow_url_fetch: bool = False, allow_local_paths: bool = False,
               max_image_mb: float = 25.0,
               max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
               max_fetch_seconds: float = DEFAULT_MAX_FETCH_SECONDS,
               load_in_8bit: bool = False, load_in_4bit: bool = False) -> None:
    """Load the model into the global worker. Call once before serving."""
    global _worker
    if dtype not in DTYPES:
        raise ValueError(f"dtype must be one of {sorted(DTYPES)}, got {dtype!r}")
    quantization = "8bit" if load_in_8bit else "4bit" if load_in_4bit else None
    _config.update({
        "model_path": model_path,
        "device": device,
        "dtype": dtype,
        "quantization": quantization,
        "allow_url_fetch": allow_url_fetch,
        "allow_local_paths": allow_local_paths,
        "max_image_bytes": int(max_image_mb * 1024 * 1024),
        "max_image_pixels": int(max_image_pixels),
        "max_fetch_seconds": float(max_fetch_seconds),
    })
    app.config["MAX_CONTENT_LENGTH"] = _body_cap(max_image_mb)
    quant_note = f", {quantization}" if quantization else ""
    print(f"[locateanything] loading {model_path} on {device} ({dtype}{quant_note}) ...", flush=True)
    _worker = LocateAnythingWorker(model_path, device=device, dtype=DTYPES[dtype],
                                   load_in_8bit=load_in_8bit, load_in_4bit=load_in_4bit)
    print("[locateanything] model ready.", flush=True)


# ---- Request helpers ----

def _request_data() -> dict:
    """Return request payload as a dict — JSON body, or form fields for multipart."""
    if request.is_json:
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise BadRequest("JSON body must be an object")
        return data
    return request.form.to_dict()


def _open_rgb(raw: bytes) -> Image.Image:
    """Decode raw image bytes into an RGB PIL image, enforcing byte/pixel caps."""
    cap = _config["max_image_bytes"]
    if cap and len(raw) > cap:
        raise BadRequest(f"image exceeds the {cap // (1024 * 1024)} MB limit")
    # Parse the header first; img.size is known before load() allocates pixels,
    # so we can reject a decompression bomb (small file, huge canvas) up front.
    try:
        img = Image.open(io.BytesIO(raw))
        width, height = img.size
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise BadRequest(f"could not decode image: {exc}")
    px_cap = _config["max_image_pixels"]
    if px_cap and width * height > px_cap:
        raise BadRequest(
            f"image {width}x{height} ({width * height} px) exceeds the {px_cap} pixel limit"
        )
    try:
        img.load()
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise BadRequest(f"could not decode image: {exc}")
    return img.convert("RGB")


def _fetch_bytes(url: str) -> bytes:
    """Read image bytes from an http(s) URL or local path (both opt-in)."""
    if url.startswith(("http://", "https://")):
        if not _config["allow_url_fetch"]:
            raise BadRequest("remote URL fetch is disabled; start with --allow-url-fetch")
        import requests  # lazy: only needed when URL fetch is enabled
        resp = requests.get(url, timeout=15, stream=True)
        resp.raise_for_status()
        cap = _config["max_image_bytes"]
        # requests' scalar timeout is per-socket-op, not total — add a wall clock
        # so a slow-drip server can't pin the worker thread indefinitely.
        budget = _config["max_fetch_seconds"]
        deadline = time.monotonic() + budget
        chunks, total = [], 0
        for chunk in resp.iter_content(64 * 1024):
            if time.monotonic() > deadline:
                resp.close()
                raise BadRequest(f"image fetch exceeded the {budget}s time limit")
            total += len(chunk)
            if cap and total > cap:
                raise BadRequest(f"image exceeds the {cap // (1024 * 1024)} MB limit")
            chunks.append(chunk)
        return b"".join(chunks)
    # Anything else is treated as a local filesystem path.
    if not _config["allow_local_paths"]:
        raise BadRequest("local path reads are disabled; start with --allow-local-paths")
    try:
        with open(url, "rb") as handle:
            return handle.read()
    except OSError as exc:
        raise BadRequest(f"could not read image path: {exc}")


def _load_image(data: dict) -> Image.Image:
    """Extract a PIL RGB image from the request (file upload, base64, or url/path)."""
    upload = request.files.get("image") or request.files.get("file")
    if upload is not None and upload.filename:
        return _open_rgb(upload.read())

    b64 = data.get("image")
    if isinstance(b64, str) and b64:
        if b64.startswith("data:"):
            b64 = b64.split(",", 1)[-1]
        try:
            raw = base64.b64decode(b64, validate=True)
        except (binascii.Error, ValueError):
            raise BadRequest("'image' field is not valid base64")
        return _open_rgb(raw)

    url = data.get("image_url")
    if isinstance(url, str) and url:
        return _open_rgb(_fetch_bytes(url))

    raise BadRequest(
        "no image provided — use a multipart 'image' file, a base64 'image' "
        "string, or an 'image_url'"
    )


def _as_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _gen_kwargs(data: dict) -> dict:
    """Pull optional generation parameters out of the request payload."""
    kwargs = {}
    if data.get("generation_mode") not in (None, ""):
        mode = str(data["generation_mode"]).lower()
        if mode not in VALID_MODES:
            raise BadRequest(f"generation_mode must be one of {sorted(VALID_MODES)}")
        kwargs["generation_mode"] = mode
    if data.get("max_new_tokens") not in (None, ""):
        try:
            kwargs["max_new_tokens"] = int(data["max_new_tokens"])
        except (TypeError, ValueError):
            raise BadRequest("max_new_tokens must be an integer")
    if data.get("temperature") not in (None, ""):
        try:
            kwargs["temperature"] = float(data["temperature"])
        except (TypeError, ValueError):
            raise BadRequest("temperature must be a number")
    if data.get("verbose") not in (None, ""):
        kwargs["verbose"] = _as_bool(data["verbose"])
    return kwargs


def _json_safe(value):
    """Best-effort coercion of arbitrary stats into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, torch.Tensor):
        return value.tolist() if value.numel() <= 64 else f"<tensor {tuple(value.shape)}>"
    return str(value)


def _run(call: Callable[[LocateAnythingWorker], dict]) -> dict:
    """Run an inference call against the shared worker under the GPU lock."""
    if _worker is None:
        raise ServiceUnavailable("model is not loaded yet")
    with _infer_lock:
        return call(_worker)


def _respond(image: Image.Image, result: dict):
    """Build the standard JSON response: raw answer + parsed boxes/points."""
    width, height = image.size
    answer = result.get("answer", "")
    payload = {
        "answer": answer,
        "boxes": LocateAnythingWorker.parse_boxes(answer, width, height),
        "points": LocateAnythingWorker.parse_points(answer, width, height),
        "image_size": {"width": width, "height": height},
    }
    if "stats" in result:
        payload["stats"] = _json_safe(result["stats"])
    return jsonify(payload)


def _require_phrase(data: dict) -> str:
    phrase = data.get("phrase") or data.get("text")
    if not isinstance(phrase, str) or not phrase.strip():
        raise BadRequest("'phrase' text field is required")
    return phrase


# ---- Routes ----

@app.get("/")
def index():
    """Serve the minimal upload-and-visualize web UI."""
    return send_from_directory(app.static_folder, "index.html")


@app.get("/api")
def api_index():
    """Machine-readable listing of the inference endpoints."""
    return jsonify({
        "service": "LocateAnything",
        "model_path": _config["model_path"],
        "endpoints": [
            "GET  /             web UI",
            "GET  /health",
            "POST /predict {image, question}",
            "POST /detect {image, categories}",
            "POST /ground_single {image, phrase}",
            "POST /ground_multi {image, phrase}",
            "POST /ground_text {image, phrase}",
            "POST /detect_text {image}",
            "POST /ground_gui {image, phrase, output_type=box|point}",
            "POST /point {image, phrase}",
        ],
    })


@app.get("/health")
def health():
    loaded = _worker is not None
    payload = {
        "status": "ok" if loaded else "loading",
        "model_loaded": loaded,
        "model_path": _config["model_path"],
        "device": _config["device"],
        "dtype": _config["dtype"],
        "quantization": _config["quantization"],
    }
    # 503 until the model is loaded so readiness probes don't route traffic early.
    return jsonify(payload), (200 if loaded else 503)


@app.post("/predict")
def predict():
    data = _request_data()
    image = _load_image(data)
    question = data.get("question") or data.get("prompt")
    if not isinstance(question, str) or not question.strip():
        raise BadRequest("'question' (or 'prompt') text field is required")
    gen = _gen_kwargs(data)
    result = _run(lambda w: w.predict(image, question, **gen))
    return _respond(image, result)


@app.post("/detect")
def detect():
    data = _request_data()
    image = _load_image(data)
    categories = data.get("categories")
    if isinstance(categories, str):
        categories = [c.strip() for c in categories.split(",") if c.strip()]
    if not isinstance(categories, list) or not categories:
        raise BadRequest("'categories' (list or comma-separated string) is required")
    categories = [str(c) for c in categories]
    gen = _gen_kwargs(data)
    result = _run(lambda w: w.detect(image, categories, **gen))
    return _respond(image, result)


@app.post("/ground_single")
def ground_single():
    data = _request_data()
    image = _load_image(data)
    phrase = _require_phrase(data)
    gen = _gen_kwargs(data)
    result = _run(lambda w: w.ground_single(image, phrase, **gen))
    return _respond(image, result)


@app.post("/ground_multi")
def ground_multi():
    data = _request_data()
    image = _load_image(data)
    phrase = _require_phrase(data)
    gen = _gen_kwargs(data)
    result = _run(lambda w: w.ground_multi(image, phrase, **gen))
    return _respond(image, result)


@app.post("/ground_text")
def ground_text():
    data = _request_data()
    image = _load_image(data)
    phrase = _require_phrase(data)
    gen = _gen_kwargs(data)
    result = _run(lambda w: w.ground_text(image, phrase, **gen))
    return _respond(image, result)


@app.post("/detect_text")
def detect_text():
    data = _request_data()
    image = _load_image(data)
    gen = _gen_kwargs(data)
    result = _run(lambda w: w.detect_text(image, **gen))
    return _respond(image, result)


@app.post("/ground_gui")
def ground_gui():
    data = _request_data()
    image = _load_image(data)
    phrase = _require_phrase(data)
    output_type = str(data.get("output_type", "box")).lower()
    if output_type not in {"box", "point"}:
        raise BadRequest("output_type must be 'box' or 'point'")
    gen = _gen_kwargs(data)
    result = _run(lambda w: w.ground_gui(image, phrase, output_type=output_type, **gen))
    return _respond(image, result)


@app.post("/point")
def point():
    data = _request_data()
    image = _load_image(data)
    phrase = _require_phrase(data)
    gen = _gen_kwargs(data)
    result = _run(lambda w: w.point(image, phrase, **gen))
    return _respond(image, result)


# ---- Error handling ----

@app.errorhandler(HTTPException)
def _on_http_error(exc: HTTPException):
    return jsonify({"error": exc.description, "status_code": exc.code}), exc.code


@app.errorhandler(Exception)
def _on_error(exc: Exception):
    app.logger.exception("inference failed")
    return jsonify({"error": str(exc), "status_code": 500}), 500


# ---- Entry points ----

def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    return default if raw is None else _as_bool(raw)


def main():
    parser = argparse.ArgumentParser(description="Serve LocateAnything over HTTP (Flask).")
    parser.add_argument("--model-path", default=os.environ.get("LA_MODEL_PATH", "nvidia/LocateAnything-3B"))
    parser.add_argument("--host", default=os.environ.get("LA_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("LA_PORT", "8000")))
    parser.add_argument("--device", default=os.environ.get("LA_DEVICE", "cuda"))
    parser.add_argument("--dtype", default=os.environ.get("LA_DTYPE", "bfloat16"), choices=sorted(DTYPES))
    parser.add_argument("--allow-url-fetch", action="store_true", default=_env_bool("LA_ALLOW_URL_FETCH", False),
                        help="permit fetching images from http(s) URLs (SSRF risk)")
    parser.add_argument("--allow-local-paths", action="store_true", default=_env_bool("LA_ALLOW_LOCAL_PATHS", False),
                        help="permit reading images from local filesystem paths")
    parser.add_argument("--max-image-mb", type=float, default=DEFAULT_MAX_IMAGE_MB,
                        help="reject request bodies / images larger than this many MB")
    parser.add_argument("--max-image-pixels", type=int, default=DEFAULT_MAX_IMAGE_PIXELS,
                        help="reject images whose width*height exceeds this (bomb guard)")
    parser.add_argument("--max-fetch-seconds", type=float, default=DEFAULT_MAX_FETCH_SECONDS,
                        help="wall-clock limit for --allow-url-fetch downloads")
    parser.add_argument("--load-in-8bit", action="store_true", default=_env_bool("LA_LOAD_IN_8BIT", False),
                        help="load weights as int8 (bitsandbytes) to roughly halve GPU memory")
    parser.add_argument("--load-in-4bit", action="store_true", default=_env_bool("LA_LOAD_IN_4BIT", False),
                        help="load weights as 4-bit NF4 (bitsandbytes); smallest footprint")
    args = parser.parse_args()

    load_model(args.model_path, device=args.device, dtype=args.dtype,
               allow_url_fetch=args.allow_url_fetch, allow_local_paths=args.allow_local_paths,
               max_image_mb=args.max_image_mb, max_image_pixels=args.max_image_pixels,
               max_fetch_seconds=args.max_fetch_seconds,
               load_in_8bit=args.load_in_8bit, load_in_4bit=args.load_in_4bit)
    # threaded=True keeps /health responsive during inference; the GPU lock
    # serializes the actual model calls. Use one process (the model is large).
    app.run(host=args.host, port=args.port, threaded=True)


# When launched via a WSGI server (e.g. gunicorn) main() is never called, so
# load the model at import time if LA_MODEL_PATH is set. Skipped under the CLI
# entry point (__main__), where main() does the loading — avoids a double load.
if __name__ != "__main__" and _worker is None and os.environ.get("LA_MODEL_PATH"):
    load_model(
        os.environ["LA_MODEL_PATH"],
        device=os.environ.get("LA_DEVICE", "cuda"),
        dtype=os.environ.get("LA_DTYPE", "bfloat16"),
        allow_url_fetch=_env_bool("LA_ALLOW_URL_FETCH", False),
        allow_local_paths=_env_bool("LA_ALLOW_LOCAL_PATHS", False),
        max_image_mb=DEFAULT_MAX_IMAGE_MB,
        max_image_pixels=DEFAULT_MAX_IMAGE_PIXELS,
        max_fetch_seconds=DEFAULT_MAX_FETCH_SECONDS,
        load_in_8bit=_env_bool("LA_LOAD_IN_8BIT", False),
        load_in_4bit=_env_bool("LA_LOAD_IN_4BIT", False),
    )


if __name__ == "__main__":
    main()
