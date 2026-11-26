"""
data_tools/coco_dataset.py
==========================
Loader for COCO-format detection annotations plus reproducible
train / val / test splitting.

NO DATASET IS INCLUDED WITH THIS PROJECT.  Download one yourself (see
DATASETS.md), convert it to COCO JSON if needed, and point the scripts at it.

COCO JSON layout expected:
  {"images": [{"id", "file_name", "width", "height"}],
   "annotations": [{"image_id", "category_id", "bbox": [x, y, w, h]}],
   "categories": [{"id", "name"}]}

Only vehicle classes are kept: car, motorcycle, bus, truck (COCO IDs 2, 3, 5, 7).
Categories are matched by NAME, so datasets with different numeric IDs
(e.g. BDD100K exported to COCO) are mapped correctly.
"""
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

VEHICLE_NAME_TO_ID = {"car": 2, "motorcycle": 3, "bus": 5, "truck": 7}


@dataclass
class GroundTruthImage:
    image_id: int
    file_name: str
    width: int
    height: int
    # (x1, y1, x2, y2, class_id) in pixels
    boxes: List[Tuple[float, float, float, float, int]] = field(default_factory=list)


def load_coco_vehicles(annotation_path: str) -> Dict[int, GroundTruthImage]:
    """Load a COCO JSON file, keeping only vehicle boxes. Returns {image_id: GroundTruthImage}."""
    if not os.path.isfile(annotation_path):
        raise FileNotFoundError(f"Annotation file not found: {annotation_path}")
    with open(annotation_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    for key in ("images", "annotations", "categories"):
        if key not in data:
            raise ValueError(f"COCO JSON missing required key '{key}'")

    cat_to_vehicle: Dict[int, int] = {}
    for c in data["categories"]:
        name = str(c.get("name", "")).strip().lower()
        if name in VEHICLE_NAME_TO_ID:
            cat_to_vehicle[c["id"]] = VEHICLE_NAME_TO_ID[name]
    if not cat_to_vehicle:
        raise ValueError(
            "No vehicle categories (car/motorcycle/bus/truck) found in 'categories'."
        )

    images = {
        im["id"]: GroundTruthImage(im["id"], im["file_name"], im.get("width", 0), im.get("height", 0))
        for im in data["images"]
    }
    skipped = 0
    for a in data["annotations"]:
        cid = cat_to_vehicle.get(a["category_id"])
        if cid is None or a["image_id"] not in images:
            skipped += 1
            continue
        x, y, w, h = a["bbox"]
        if w <= 0 or h <= 0:
            skipped += 1
            continue
        images[a["image_id"]].boxes.append((float(x), float(y), float(x + w), float(y + h), cid))
    logger.info("Loaded %d images; skipped %d non-vehicle/invalid annotations.", len(images), skipped)
    return images


def make_splits(
    image_ids: List[int],
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> Dict[str, List[int]]:
    """Reproducible, disjoint train/val/test split of image IDs."""
    if len(ratios) != 3 or any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must be 3 non-negative numbers summing to 1; got {ratios}")
    ids = sorted(set(image_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(round(ratios[0] * n))
    n_val = int(round(ratios[1] * n))
    return {
        "train": sorted(ids[:n_train]),
        "val": sorted(ids[n_train:n_train + n_val]),
        "test": sorted(ids[n_train + n_val:]),
    }


def save_splits(splits: Dict[str, List[int]], path: str, seed: Optional[int] = None) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"seed": seed, "splits": splits}, f, indent=2)


def load_splits(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["splits"]
