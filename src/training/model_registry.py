from __future__ import annotations

import shutil
from pathlib import Path


def register_best_model(best_weights: str, target: str = "models/flowtrack_best.pt") -> Path:
    src = Path(best_weights)
    if not src.exists():
        raise FileNotFoundError(f"Weights not found: {src}")

    dst = Path(target)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return dst
