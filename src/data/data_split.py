"""
Leakage-Free Data Splitting Module.
Ensures source manuscripts/documents and all their synthetic variants remain
strictly isolated across Train, Validation, and Test splits.
"""
from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import random
from ..core.config import settings


def split_document_ids(
    document_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Splits unique document IDs into train, validation, and test sets.
    All crops, augmentations, and synthetic degradations of the same document ID
    MUST inherit this split assignment to prevent data leakage.
    """
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-5, "Ratios must sum to 1.0"

    rng = random.Random(seed)
    shuffled = sorted(list(set(document_ids)))
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
        "counts": {
            "total": n_total,
            "train": len(train_ids),
            "val": len(val_ids),
            "test": len(test_ids),
        },
        "seed": seed,
    }


def save_split_manifest(split_dict: Dict[str, Any], output_path: Path = None) -> Path:
    """Saves split manifest to JSON for reproducibility."""
    if output_path is None:
        output_path = settings.SPLITS_DIR / "split_manifest.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(split_dict, f, indent=2)

    return output_path


def load_split_manifest(manifest_path: Path = None) -> Dict[str, Any]:
    """Loads existing split manifest."""
    if manifest_path is None:
        manifest_path = settings.SPLITS_DIR / "split_manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Split manifest not found at {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)
