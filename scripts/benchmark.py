# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""
benchmark.py - Repeatable speed benchmark for LocateAnything.

Compares attention backends (sdpa vs flash_attention_2) and optional
``torch.compile`` targets across generation modes, using the model's own
per-call stats (tps / bps / prefill_time / forward_step) plus an independent
CUDA-synchronized wall clock.

Example:
    PYTHONNOUSERSITE=1 python scripts/benchmark.py \
        --configs sdpa:none flash_attention_2:none flash_attention_2:both \
        --modes fast hybrid --runs 3 --warmup 1 \
        --out doc/bench_results.md
"""
import argparse
import glob
import os
import re
import statistics
import sys
import time

import torch
from PIL import Image

# Run directly without an editable install.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from locateanything import LocateAnythingWorker  # noqa: E402

DEFAULT_IMAGES = [
    "/home/gc635/Documents/eagle/Embodied/assets/images/Smart_City.png",
    "/home/gc635/Documents/eagle/Embodied/assets/images/teaser.jpg",
    "/home/gc635/Documents/eagle/Embodied/assets/images/qualitative_examples.jpg",
]
DEFAULT_CATEGORIES = ["car", "person", "traffic light", "building", "tree"]

_STAT_RE = re.compile(r"(\w+(?:\([^)]*\))?)=([-\d.]+)")


def parse_stats(stats_str: str) -> dict:
    """Parse the worker's stats string into floats keyed by metric name."""
    out = {}
    for key, val in _STAT_RE.findall(stats_str or ""):
        key = key.split("(")[0]  # generate_time(s) -> generate_time
        try:
            out[key] = float(val)
        except ValueError:
            pass
    return out


def time_one(worker, image, categories, mode, temperature):
    """Run a single detect call; return (wall_seconds, parsed_stats, num_boxes)."""
    torch.manual_seed(0)
    if torch.cuda.is_available():
        # Free cached blocks BEFORE timing so the large (sdpa) vision-attention
        # buffer from the previous call doesn't fragment the allocator / OOM.
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = worker.detect(
        image, categories, generation_mode=mode,
        temperature=temperature, verbose=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    stats = parse_stats(out.get("stats", ""))
    n_boxes = len(LocateAnythingWorker.parse_boxes(out["answer"], *image.size))
    return wall, stats, n_boxes


def load_image(path, max_side):
    """Open as RGB and downscale so the long side <= max_side (0 disables).

    Bounds the Moon-ViT token count so the sdpa baseline (O(L^2) attention) fits
    in GPU memory; FA2 and sdpa then run on identical inputs for a fair compare.
    """
    im = Image.open(path).convert("RGB")
    if max_side and max(im.size) > max_side:
        scale = max_side / max(im.size)
        new = (round(im.width * scale), round(im.height * scale))
        im = im.resize(new, Image.BICUBIC)
    return im


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


def dynamo_counts():
    """Best-effort recompile / graph-break counters from torch._dynamo."""
    try:
        from torch._dynamo.utils import counters
        recompiles = sum(counters.get("recompiles", {}).values()) if "recompiles" in counters else 0
        graph_breaks = sum(counters.get("graph_break", {}).values()) if "graph_break" in counters else 0
        return recompiles, graph_breaks
    except Exception:
        return None, None


def run_config(attn, compile_target, args, images):
    """Load a worker for one (attn, compile) config and benchmark all modes."""
    label = f"{attn}:{compile_target}"
    print(f"\n=== loading config {label} ===", flush=True)
    # `attn` selects the VISION-encoder backend (the only tower that can vary —
    # the decoder's PBD only supports magi/sdpa, so we pin it to sdpa for a fair,
    # apples-to-apples decoder across configs).
    vision_attn = None if attn in ("default", "auto") else attn
    t_load = time.perf_counter()
    worker = LocateAnythingWorker(
        args.model, device="cuda", dtype=torch.bfloat16,
        vision_attn=vision_attn, text_attn="sdpa", compile_target=compile_target,
    )
    load_s = time.perf_counter() - t_load
    print(f"[{label}] loaded in {load_s:.1f}s (vision={worker.vision_attn}, text={worker.text_attn})",
          flush=True)

    rows = {}
    for mode in args.modes:
        walls, bps, tps, prefill, steps, boxes = [], [], [], [], [], []
        for image in images:
            for _ in range(args.warmup):
                time_one(worker, image, DEFAULT_CATEGORIES, mode, args.temperature)
            for _ in range(args.runs):
                wall, st, nb = time_one(worker, image, DEFAULT_CATEGORIES, mode, args.temperature)
                walls.append(wall)
                bps.append(st.get("bps"))
                tps.append(st.get("tps"))
                prefill.append(st.get("prefill_time"))
                steps.append(st.get("forward_step"))
                boxes.append(nb)
        rec, gb = dynamo_counts()
        rows[mode] = {
            "wall_s": median(walls), "bps": median(bps), "tps": median(tps),
            "prefill_s": median(prefill), "forward_step": median(steps),
            "boxes_mean": (sum(boxes) / len(boxes)) if boxes else float("nan"),
            "recompiles": rec, "graph_breaks": gb,
        }
        r = rows[mode]
        print(f"[{label}][{mode}] bps={r['bps']:.2f} tps={r['tps']:.1f} "
              f"prefill={r['prefill_s']:.3f}s wall={r['wall_s']:.3f}s "
              f"steps={r['forward_step']:.0f} boxes~{r['boxes_mean']:.1f} "
              f"recompiles={rec} graph_breaks={gb}", flush=True)

    del worker
    torch.cuda.empty_cache()
    return label, rows


def render_markdown(results, args):
    lines = ["# LocateAnything speed benchmark", ""]
    lines.append(f"- model: `{args.model}`")
    lines.append(f"- runs={args.runs}, warmup={args.warmup}, images={len(args.images)}, "
                 f"max_side={args.max_side}, temperature={args.temperature}")
    try:
        import transformers
        lines.append(f"- torch `{torch.__version__}`, transformers `{transformers.__version__}`, "
                     f"GPU `{torch.cuda.get_device_name(0)}`")
    except Exception:
        pass
    for mode in args.modes:
        lines += ["", f"## mode = {mode}", "",
                  "| config | bps | tps | prefill (s) | wall (s) | fwd steps | boxes~ | recompiles | graph_breaks |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for label, rows in results:
            r = rows[mode]
            lines.append(
                f"| {label} | {r['bps']:.2f} | {r['tps']:.1f} | {r['prefill_s']:.3f} | "
                f"{r['wall_s']:.3f} | {r['forward_step']:.0f} | {r['boxes_mean']:.1f} | "
                f"{r['recompiles']} | {r['graph_breaks']} |")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser(description="Benchmark LocateAnything attention/compile configs.")
    p.add_argument("--model", default="nvidia/LocateAnything-3B")
    p.add_argument("--configs", nargs="+", default=["sdpa:none", "flash_attention_2:none"],
                   help="space-separated attn:compile_target pairs, "
                        "e.g. sdpa:none flash_attention_2:none flash_attention_2:both")
    p.add_argument("--modes", nargs="+", default=["fast", "hybrid"])
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy/deterministic (recommended for benchmarking so "
                        "FA2 vs sdpa are directly comparable)")
    p.add_argument("--images", nargs="+", default=DEFAULT_IMAGES)
    p.add_argument("--max-side", type=int, default=1024,
                   help="downscale long side to this many px (0 = full res; "
                        "sdpa OOMs at full res on a 24GB card)")
    p.add_argument("--out", default=None, help="write a markdown report to this path")
    args = p.parse_args()

    args.images = [g for pat in args.images for g in (glob.glob(pat) or [pat])]
    images = [load_image(pth, args.max_side) for pth in args.images]
    print(f"images: {[(os.path.basename(p), im.size) for p, im in zip(args.images, images)]}",
          flush=True)

    results = []
    for cfg in args.configs:
        attn, _, ctarget = cfg.partition(":")
        ctarget = ctarget or "none"
        results.append(run_config(attn, ctarget, args, images))

    report = render_markdown(results, args)
    print("\n" + report)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(report)
        print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
