"""
Processor registry.

Add new processors here. They'll automatically appear in the UI.
"""

from .base import BaseProcessor, Region, DetectionResult
from .connected_components import ConnectedComponentProcessor
from .color_cluster import ColorClusterProcessor
from .superpixels import SuperpixelProcessor

# ── Register all available processors ──
# Order here = order in UI dropdown
PROCESSORS: dict[str, BaseProcessor] = {}


def _register(cls):
    instance = cls()
    PROCESSORS[cls.__name__] = instance


_register(ConnectedComponentProcessor)
_register(ColorClusterProcessor)
_register(SuperpixelProcessor)

# ── To add a new processor: ──
# 1. Create processors/my_processor.py with class MyProcessor(BaseProcessor)
# 2. Import it here
# 3. _register(MyProcessor)
# That's it — it appears in the UI automatically.


def get_processor(name: str) -> BaseProcessor:
    if name not in PROCESSORS:
        raise KeyError(f"Unknown processor: {name}. Available: {list(PROCESSORS.keys())}")
    return PROCESSORS[name]


def list_processors() -> list[dict]:
    return [
        {
            "key": key,
            "name": proc.name,
            "description": proc.description,
            "params": proc.get_all_params(),
        }
        for key, proc in PROCESSORS.items()
    ]
