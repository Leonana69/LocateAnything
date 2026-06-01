# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""
Vendored LocateAnything model code (inference only).

Importing this package registers the custom config / model / processor classes
with the HuggingFace ``Auto*`` registries, so the model can be loaded from the
published weights *without* ``trust_remote_code=True`` — the code runs from this
local package instead of being downloaded from the Hub.
"""
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
)

from .configuration_locateanything import LocateAnythingConfig, MoonViTConfig
from .image_processing_locateanything import LocateAnythingImageProcessor
from .modeling_locateanything import (
    LocateAnythingForConditionalGeneration,
    LocateAnythingPreTrainedModel,
)
from .processing_locateanything import LocateAnythingProcessor

__all__ = [
    "LocateAnythingConfig",
    "MoonViTConfig",
    "LocateAnythingImageProcessor",
    "LocateAnythingForConditionalGeneration",
    "LocateAnythingPreTrainedModel",
    "LocateAnythingProcessor",
]


def register_auto_classes() -> None:
    """Register LocateAnything with the transformers ``Auto*`` factories.

    Idempotent: safe to call multiple times. Lets ``AutoModel.from_pretrained``,
    ``AutoProcessor.from_pretrained``, etc. resolve to the vendored classes.
    """
    AutoConfig.register("moonvit", MoonViTConfig, exist_ok=True)
    AutoConfig.register("locateanything", LocateAnythingConfig, exist_ok=True)
    AutoModel.register(
        LocateAnythingConfig, LocateAnythingForConditionalGeneration, exist_ok=True
    )
    # transformers changed the AutoImageProcessor.register signature across
    # versions; try the keyword form first, fall back to positional.
    try:
        AutoImageProcessor.register(
            LocateAnythingConfig,
            slow_image_processor_class=LocateAnythingImageProcessor,
            exist_ok=True,
        )
    except TypeError:
        AutoImageProcessor.register(
            LocateAnythingConfig, LocateAnythingImageProcessor, exist_ok=True
        )
    AutoProcessor.register(LocateAnythingConfig, LocateAnythingProcessor, exist_ok=True)


register_auto_classes()
