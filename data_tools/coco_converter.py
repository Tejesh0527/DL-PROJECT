#!/usr/bin/env python3
"""
COCO Dataset Converter for RATFA

Converts various traffic detection datasets to COCO format.
Supports: BDD100K, UA-DETRAC, custom formats
"""

import argparse
import json
import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np


VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck']
CLASS_TO_ID = {name: i + 1 for i, name in enumerate(VEHICLE_CLASSES)}  # COCO 1-indexing


def convert_bdd100k_labels(labels_dir: str, output_path: str, images_dir: str) -> None:
    """Convert BDD100K labels to COCO format."""

    images = []
    annotations = []
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(VEHICLE_CLASSES)]

    annotation_files = list(Path(labels_dir).glob('*.json')) if os.path.isdir(labels_dir) else []

    ann_id = 1
    for label_file in annotation_files:
        with open(label_file, 'r') as f:
            data = json.load(f)

        img_name = data.get('name', label_file.stem + '.jpg')
        img_path = os.path.join(images_dir, img_name)

        if not os.path.exists(img_path):
            # Try without extension
            for ext in ['.jpg', '.png', '.jpeg']:
                if os.path.exists(img_path + ext):
                    img_path += ext
                    break

        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue

        h, w = img.shape[:2]
        img_id = len(images) + 1

        images.append({
            "id": img_id,
            "file_name": img_name,
            "width": w,
            "height": h
        })

        # Process vehicles
        labels = data.get('labels', [])
        for label in labels:
            if label.get('type') != 'traffic light' and 'box2d' in label:
                box2d = label['box2d']
                x1, y1 = box2d['x1'], box2d['y1']
                x2, y2 = box2d['x2'], box2d['y2']

                # Check if it's a vehicle
                category = label.get('category', '')
                if category in CLASS_TO_ID:
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": CLASS_TO_ID[category],
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "area": (x2 - x1) * (y2 - y1),
                        "iscrowd": 0
                    })
                    ann_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Converted {len(images)} images, {len(annotations)} annotations to COCO format")


def create_weather_splits(images_dir: str, condition_csv: str, output_path: str,
                         ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
                         seed: int = 42) -> None:
    """
    Create train/val/test splits with weather condition metadata.

    Args:
        condition_csv: CSV with columns: file_name, weather_condition
        output_path: Path to save splits.json
    """
    import random
    random.seed(seed)

    # Load conditions
    conditions = {}
    with open(condition_csv, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            conditions[row['file_name']] = row.get('weather_condition', 'clear')

    # Get all images
    image_extensions = ['.jpg', '.png', '.jpeg']
    all_images = []
    for f in os.listdir(images_dir):
        if any(f.lower().endswith(ext) for ext in image_extensions):
            all_images.append(f)

    # Shuffle and split
    random.shuffle(all_images)
    n = len(all_images)
    n_train = int(ratios[0] * n)
    n_val = int(ratios[1] * n)

    splits = {
        "seed": seed,
        "splits": {
            "train": all_images[:n_train],
            "val": all_images[n_train:n_train + n_val],
            "test": all_images[n_train + n_val:]
        },
        "conditions": conditions
    }

    with open(output_path, 'w') as f:
        json.dump(splits, f, indent=2)

    print(f"Created splits: {len(splits['splits']['train'])} train, "
          f"{len(splits['splits']['val'])} val, {len(splits['splits']['test'])} test")


def create_sample_dataset(output_dir: str) -> None:
    """Create a small sample dataset for testing the pipeline."""

    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    # Create sample images with synthetic vehicles
    np.random.seed(42)
    for i in range(100):
        img = np.ones((480, 640, 3), dtype=np.uint8) * 220  # Light gray background

        # Add 1-5 random "vehicles" (colored rectangles)
        num_vehicles = np.random.randint(1, 6)
        for _ in range(num_vehicles):
            w, h = np.random.randint(40, 120), np.random.randint(60, 150)
            x, y = np.random.randint(0, 640 - w), np.random.randint(0, 480 - h)

            color = tuple(np.random.randint(50, 200, 3).tolist())
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

        cv2.imwrite(os.path.join(images_dir, f'sample_{i:04d}.jpg'), img)

    # Create COCO annotations
    images = []
    annotations = []
    categories = [{"id": i + 1, "name": name} for i, name in enumerate(VEHICLE_CLASSES)]

    for i in range(100):
        img_id = i + 1
        images.append({
            "id": img_id,
            "file_name": f'sample_{i:04d}.jpg',
            "width": 640,
            "height": 480
        })

    # Create annotations for each image
    np.random.seed(42)
    ann_id = 1
    for i in range(100):
        img_id = i + 1
        num_vehicles = np.random.randint(1, 6)
        vehicle_class = np.random.choice(VEHICLE_CLASSES)

        for _ in range(num_vehicles):
            w, h = np.random.randint(40, 120), np.random.randint(60, 150)
            x, y = np.random.randint(0, 640 - w), np.random.randint(0, 480 - h)

            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": CLASS_TO_ID[vehicle_class],
                "bbox": [float(x), float(y), float(w), float(h)],
                "area": float(w * h),
                "iscrowd": 0
            })
            ann_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations,
        "categories": categories
    }

    with open(os.path.join(output_dir, 'instances_sample.json'), 'w') as f:
        json.dump(coco_data, f, indent=2)

    print(f"Created sample dataset: 100 images, {len(annotations)} annotations")


def main():
    parser = argparse.ArgumentParser(description='Convert datasets to COCO format for RATFA')
    parser.add_argument('--bdd-labels', default=None, help='BDD100K labels directory')
    parser.add_argument('--bdd-images', default=None, help='BDD100K images directory')
    parser.add_argument('--output', default='data/bdd100k_coco.json', help='Output COCO JSON path')
    parser.add_argument('--condition-csv', default=None, help='CSV with weather conditions')
    parser.add_argument('--make-splits', action='store_true', help='Also create train/val/test splits')
    parser.add_argument('--create-sample', action='store_true', help='Create synthetic sample dataset')
    parser.add_argument('--sample-dir', default='data/sample_dataset', help='Output for sample dataset')
    args = parser.parse_args()

    if args.create_sample:
        create_sample_dataset(args.sample_dir)
        return

    if args.bdd_labels and args.bdd_images:
        convert_bdd100k_labels(args.bdd_labels, args.output, args.bdd_images)

    if args.condition_csv and args.make_splits:
        output_path = args.output.replace('.json', '_splits.json')
        create_weather_splits(args.bdd_images, args.condition_csv, output_path)


if __name__ == '__main__':
    main()