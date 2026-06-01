# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
worker.py - A reusable worker for LocateAnything inference.

The model code is vendored locally (see ``locateanything.model``); importing it
registers the custom classes with the transformers ``Auto*`` registries, so the
weights load *without* ``trust_remote_code=True``.
"""
import importlib.util
import re

import torch
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor, AutoTokenizer
from transformers.utils import is_flash_attn_2_available

# Importing the model package registers LocateAnything with the Auto* factories.
from . import model as _model  # noqa: F401


def _magi_available() -> bool:
    return importlib.util.find_spec("magi_attention") is not None


def _single_device_map(device: str) -> dict:
    """A device_map that pins the whole model to one device — what bitsandbytes needs."""
    dev = torch.device(device)
    target = (dev.index if dev.index is not None else 0) if dev.type == "cuda" else dev.type
    return {"": target}


def _build_quantization_config(load_in_8bit: bool, load_in_4bit: bool, dtype, device: str):
    """Build a BitsAndBytesConfig for int8/nf4 loading, or None for full precision.

    Quantization is opt-in; when neither flag is set the model loads in ``dtype`` as
    before. int8/4bit need a CUDA device and the ``bitsandbytes`` package.
    """
    if load_in_8bit and load_in_4bit:
        raise ValueError("Choose at most one of load_in_8bit / load_in_4bit, not both.")
    if not (load_in_8bit or load_in_4bit):
        return None
    if torch.device(device).type != "cuda":
        raise ValueError("8-bit / 4-bit quantization requires a CUDA device.")
    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            "8-bit / 4-bit loading needs the 'bitsandbytes' package "
            "(pip install bitsandbytes)."
        )
    from transformers import BitsAndBytesConfig

    if load_in_8bit:
        # LLM.int8(): linear weights stored in int8, outliers + activations kept in
        # fp16. Roughly halves weight memory with negligible quality loss; the vision
        # encoder, connector and lm_head still compute in higher precision.
        return BitsAndBytesConfig(load_in_8bit=True)
    # 4-bit NF4 with double quantization: ~4x smaller weights, matmuls run in `dtype`.
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=dtype,
    )


class LocateAnythingWorker:
    """Stateful worker that loads the model once and serves perception queries."""

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype=torch.bfloat16,
        vision_attn: str | None = None,
        text_attn: str | None = None,
        compile_target: str = "none",
        compile_mode: str = "default",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """
        Args:
            model_path: HF id or local dir with the weights.
            device: torch device (e.g. "cuda").
            dtype: model dtype; FA2 requires bfloat16/float16.
            vision_attn: attention backend for the Moon-ViT encoder
                ("flash_attention_2" | "sdpa" | "eager"). ``None`` -> FA2 if
                available, else sdpa.
            text_attn: attention backend for the Qwen2 decoder. NOTE: the decoder's
                Parallel Box Decoding only supports "magi" or "sdpa" — there is no
                flash_attention_2 path (see modeling_qwen2.py:1321-1335). ``None``
                -> "magi" if the ``magi_attention`` package is importable, else
                "sdpa". (Do not force "flash_attention_2" here; it raises.)
            compile_target: which submodule(s) to ``torch.compile``:
                "none" | "vision" | "llm" | "both". Experimental — the dynamic
                shapes / PBD loop may trigger recompiles or graph breaks.
            compile_mode: ``torch.compile`` mode ("default", "reduce-overhead", ...).
            load_in_8bit: load linear weights as int8 (bitsandbytes LLM.int8()).
                ~halves weight memory; CUDA + ``bitsandbytes`` required. Box accuracy
                is preserved in practice. Mutually exclusive with ``load_in_4bit``.
            load_in_4bit: load weights as 4-bit NF4 (bitsandbytes). ~4x smaller
                weights, slightly larger quality hit than int8.
        """
        self.device = device
        self.dtype = dtype

        # Resolve per-tower attention. The decoder must stay on magi/sdpa; with
        # flash-attn installed the model's own default would pick FA2 for the
        # decoder and crash, so we pin a supported impl here.
        vision_attn = vision_attn or ("flash_attention_2" if is_flash_attn_2_available() else "sdpa")
        text_attn = text_attn or ("magi" if _magi_available() else "sdpa")

        config = AutoConfig.from_pretrained(model_path)
        config.vision_config._attn_implementation = vision_attn
        config.text_config._attn_implementation = text_attn
        self.vision_attn = vision_attn
        self.text_attn = text_attn

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.processor = AutoProcessor.from_pretrained(model_path)

        quantization_config = _build_quantization_config(load_in_8bit, load_in_4bit, dtype, device)
        self.quantization = "8bit" if load_in_8bit else "4bit" if load_in_4bit else None
        if quantization_config is None:
            self.model = (
                AutoModel.from_pretrained(model_path, config=config, dtype=dtype)
                .to(device).eval()
            )
        else:
            # bitsandbytes places the quantized weights on the device itself via
            # device_map; calling .to() on a quantized model raises, so we never move
            # it afterwards. `dtype` still governs the non-quantized modules.
            self.model = AutoModel.from_pretrained(
                model_path,
                config=config,
                dtype=dtype,
                quantization_config=quantization_config,
                device_map=_single_device_map(device),
            ).eval()

        self._maybe_compile(compile_target, compile_mode)

    def _maybe_compile(self, compile_target: str, compile_mode: str) -> None:
        """Optionally ``torch.compile`` the vision and/or language submodules.

        Wrapped in try/except so a compile failure never breaks inference — on
        error we log and keep the eager module.
        """
        if compile_target == "none":
            return
        targets = {
            "vision": ["vision_model"],
            "llm": ["language_model"],
            "both": ["vision_model", "language_model"],
        }.get(compile_target)
        if targets is None:
            raise ValueError(
                f"compile_target must be none|vision|llm|both, got {compile_target!r}"
            )
        for name in targets:
            module = getattr(self.model, name, None)
            if module is None:
                continue
            try:
                compiled = torch.compile(module, mode=compile_mode, dynamic=True)
                setattr(self.model, name, compiled)
                print(f"[locateanything] torch.compile enabled on {name} "
                      f"(mode={compile_mode}, dynamic=True)", flush=True)
            except Exception as exc:  # pragma: no cover - environment dependent
                print(f"[locateanything] torch.compile on {name} failed "
                      f"({exc}); using eager.", flush=True)

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        question: str,
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        temperature: float = 0.7,
        verbose: bool = True,
    ) -> dict:
        """
        Run a single perception query.

        Args:
            image: PIL Image (RGB).
            question: The task prompt (see supported prompts below).
            generation_mode: "fast" (MTP) | "slow" (NTP) | "hybrid".
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (0 = greedy).
            verbose: If True, return timing statistics.

        Returns:
            dict with keys: "answer", "stats" (optional), "history" (optional).
        """
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": question},
            ]}
        ]

        text = self.processor.py_apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        images, videos = self.processor.process_vision_info(messages)
        inputs = self.processor(
            text=[text], images=images, videos=videos, return_tensors="pt"
        ).to(self.device)

        pixel_values = inputs["pixel_values"].to(self.dtype)
        input_ids = inputs["input_ids"]
        image_grid_hws = inputs.get("image_grid_hws", None)

        response = self.model.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=inputs["attention_mask"],
            image_grid_hws=image_grid_hws,
            tokenizer=self.tokenizer,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            generation_mode=generation_mode,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            repetition_penalty=1.1,
            verbose=verbose,
        )

        result = {"answer": response[0] if isinstance(response, tuple) else response}
        if isinstance(response, tuple) and len(response) >= 3:
            result["history"] = response[1]
            result["stats"] = response[2]
        return result

    # ---- Convenience methods for each task ----

    def detect(self, image: Image.Image, categories: list[str], **kwargs) -> dict:
        """Object detection / document layout analysis."""
        cats = "</c>".join(categories)
        prompt = f"Locate all the instances that matches the following description: {cats}."
        return self.predict(image, prompt, **kwargs)

    def ground_single(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — single instance."""
        prompt = f"Locate a single instance that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_multi(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Phrase grounding — multiple instances."""
        prompt = f"Locate all the instances that match the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def ground_text(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Text grounding."""
        prompt = f"Please locate the text referred as {phrase}."
        return self.predict(image, prompt, **kwargs)

    def detect_text(self, image: Image.Image, **kwargs) -> dict:
        """Scene text detection."""
        prompt = "Detect all the text in box format."
        return self.predict(image, prompt, **kwargs)

    def ground_gui(self, image: Image.Image, phrase: str, output_type: str = "box", **kwargs) -> dict:
        """GUI grounding (box or point)."""
        if output_type == "point":
            prompt = f"Point to: {phrase}."
        else:
            prompt = f"Locate the region that matches the following description: {phrase}."
        return self.predict(image, prompt, **kwargs)

    def point(self, image: Image.Image, phrase: str, **kwargs) -> dict:
        """Pointing."""
        prompt = f"Point to: {phrase}."
        return self.predict(image, prompt, **kwargs)

    # ---- Utility: parse model output ----

    @staticmethod
    def parse_boxes(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate bounding boxes.

        Coordinates in model output are normalized integers in [0, 1000].
        """
        boxes = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = [int(g) for g in m.groups()]
            boxes.append({
                "x1": x1 / 1000 * image_width,
                "y1": y1 / 1000 * image_height,
                "x2": x2 / 1000 * image_width,
                "y2": y2 / 1000 * image_height,
            })
        return boxes

    @staticmethod
    def parse_points(answer: str, image_width: int, image_height: int) -> list[dict]:
        """Parse model output into pixel-coordinate points."""
        points = []
        for m in re.finditer(r"<box><(\d+)><(\d+)></box>", answer):
            x, y = int(m.group(1)), int(m.group(2))
            points.append({
                "x": x / 1000 * image_width,
                "y": y / 1000 * image_height,
            })
        return points
