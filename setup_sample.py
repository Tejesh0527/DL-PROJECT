#!/usr/bin/env python3
"""
Setup script to create sample dataset and train/test the RATFA pipeline
"""

import os
import sys
import json
import subprocess

import numpy as np
import cv2


def create_sample_dataset(data_dir='data/sample_dataset'):
    """Create a sample traffic dataset for testing."""
    print("Creating sample traffic dataset...")

    images_dir = os.path.join(data_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)

    np.random.seed(42)
    annotations = {'images': [], 'annotations': [], 'categories': []}

    # Vehicle classes (COCO format)
    vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']
    class_to_id = {name: i + 1 for i, name in enumerate(vehicle_classes)}

    for i, name in enumerate(vehicle_classes):
        annotations['categories'].append({'id': i + 1, 'name': name})

    ann_id = 1
    for img_idx in range(100):
        # Create synthetic traffic image
        img = np.ones((480, 640, 3), dtype=np.uint8) * 200  # Light background

        # Add realistic elements
        # Road markings (white lines)
        cv2.line(img, (0, 400), (640, 400), (255, 255, 255), 3)

        # Sky gradient
        for y in range(200):
            shade = int(180 + y * 0.2)
            img[y, :, :] = [shade, shade + 20, shade + 40]

        # Add vehicles
        num_vehicles = np.random.randint(1, 5)
        vehicle_class = np.random.choice(vehicle_classes)

        for _ in range(num_vehicles):
            # Random vehicle size
            w, h = np.random.randint(50, 150), np.random.randint(80, 200)
            x, y = np.random.randint(100, 540 - w), np.random.randint(250, 420 - h)

            # Vehicle color
            color = (np.random.randint(50, 150),
                    np.random.randint(50, 150),
                    np.random.randint(50, 150))

            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)

            # Add wheels
            for wx in [x + 15, x + w - 15]:
                for wy in [y + h - 15, y + h - 5]:
                    cv2.circle(img, (wx, wy), 8, (30, 30, 30), -1)

        # Save image
        filename = f'sample_{img_idx:04d}.jpg'
        cv2.imwrite(os.path.join(images_dir, filename), img)

        # Add image annotation
        annotations['images'].append({
            'id': img_idx + 1,
            'file_name': filename,
            'width': 640,
            'height': 480
        })

        # Add vehicle annotations
        for _ in range(num_vehicles):
            w, h = np.random.randint(50, 150), np.random.randint(80, 200)
            x, y = np.random.randint(100, 540 - w), np.random.randint(250, 420 - h)

            annotations['annotations'].append({
                'id': ann_id,
                'image_id': img_idx + 1,
                'category_id': class_to_id[vehicle_class],
                'bbox': [float(x), float(y), float(w), float(h)],
                'area': float(w * h),
                'iscrowd': 0
            })
            ann_id += 1

    # Save COCO annotations
    with open(os.path.join(data_dir, 'instances_sample.json'), 'w') as f:
        json.dump(annotations, f, indent=2)

    print(f"Created {len(annotations['images'])} images with {len(annotations['annotations'])} vehicle annotations")
    return annotations


def run_training(annotations, images_dir, output_dir, epochs=10, batch_size=8):
    """Run training with sample dataset."""
    print("\n" + "=" * 60)
    print("TRAINING RATFA DETECTOR")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    # Create training script command
    cmd = [
        sys.executable, 'train.py',
        '--annotations', annotations,
        '--images-dir', images_dir,
        '--output-dir', output_dir,
        '--epochs', str(epochs),
        '--batch-size', str(batch_size),
        '--seed', '42'
    ]

    print(f"Running: {' '.join(cmd)}")

    # Run training
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print(result.stdout)
        print("✅ Training completed successfully!")

        # Save fuzzy parameters
        checkpoint = os.path.join(output_dir, 'checkpoints', 'best_model.pt')
        if os.path.exists(checkpoint):
            print(f"\nModel checkpoint saved: {checkpoint}")

        fuzzy_params = os.path.join(output_dir, 'fuzzy_params.json')
        if os.path.exists(fuzzy_params):
            with open(fuzzy_params) as f:
                params = json.load(f)
            print(f"Fuzzy parameters saved: {fuzzy_params}")
            print(f"FFM centers: {params['ffm_centers']}")
            print(f"FA centers: {params['fa_centers']}")
    else:
        print("Training output:")
        print(result.stdout)
        print(result.stderr)
        print("❌ Training failed")

    return result.returncode == 0


def main():
    # Paths
    project_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(project_dir, 'data', 'sample_dataset')
    output_dir = os.path.join(project_dir, 'outputs', 'ratfa_training')

    os.chdir(project_dir)

    print("=" * 60)
    print("RATFA COMPLETE SETUP AND TRAINING")
    print("=" * 60)

    # Step 1: Create sample dataset
    annotations = create_sample_dataset(data_dir)

    # Step 2: Run training
    annotations_file = os.path.join(data_dir, 'instances_sample.json')
    images_dir = os.path.join(data_dir, 'images')

    success = run_training(annotations_file, images_dir, output_dir, epochs=10)

    if success:
        print("\n" + "=" * 60)
        print("✅ RATFA SETUP COMPLETE")
        print("=" * 60)
        print(f"\nTrained model: {output_dir}/checkpoints/best_model.pt")
        print(f"Fuzzy parameters: {output_dir}/fuzzy_params.json")
        print(f"\nNext step: Run inference with main.py")
        return 0
    else:
        print("\n❌ Setup failed")
        return 1


if __name__ == '__main__':
    sys.exit(main())