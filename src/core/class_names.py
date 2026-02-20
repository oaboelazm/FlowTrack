from __future__ import annotations

from typing import Dict

ALIASES: Dict[str, str] = {
    "pedestrian": "person",
    "people": "person",
    "rider": "person",
    "person": "person",
    "bike": "bicycle",
    "bicycle": "bicycle",
    "car": "car",
    "van": "car",
    "motor": "motorcycle",
    "motorbike": "motorcycle",
    "motorcycle": "motorcycle",
    "bus": "bus",
    "truck": "truck",
    "lorry": "truck",
}


def canonical_class_name(name: str) -> str:
    key = str(name).strip().lower()
    return ALIASES.get(key, key)
