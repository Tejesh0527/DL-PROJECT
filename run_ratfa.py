#!/usr/bin/env python3
"""
RATFA Complete Pipeline Runner

Run the full Reliability-Aware Temporal Fuzzy Adaptation pipeline:
1. Train YOLO-based detector
2. Apply RATFA adaptation
3. Evaluate results
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

import yaml


def run_training(args):
    """Run model training."""
    print("=" * 60)
    print("STEP 1: TRAINING DETECTOR")
    print("=" * 60)

    cmd = [
        sys.executable, 'train.py',
        '--annotations', args.annotations,
        '--images-dir', args.images_dir,
        '--output-dir', args.output_dir,
        '--epochs', str(args.epochs),
        '--batch-size', str(args.batch_size),
        '--lr', str(args.lr),
        '--seed', str(args.seed)
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=args.project_dir)
    return result.returncode == 0


def run_evaluation(args):
    """Run model evaluation."""
    print("\n" + "=" * 60)
    print("STEP 2: EVALUATION")
    print("=" * 60)

    splits_file = os.path.join(args.output_dir, 'splits.json')
    if not os.path.exists(splits_file):
        # Create simple splits
        cmd = [
            sys.executable, 'train.py',
            '--annotations', args.annotations,
            '--images-dir', args.images_dir,
            '--make-splits', os.path.join(args.output_dir, 'splits.json'),
            '--seed', str(args.seed)
        ]
        subprocess.run(cmd, cwd=args.project_dir)

    cmd = [
        sys.executable, 'evaluate.py',
        '--config', 'config/config.yaml',
        '--annotations', args.annotations,
        '--images-dir', args.images_dir,
        '--split-file', splits_file,
        '--split', 'test',
        '--output', os.path.join(args.output_dir, 'evaluation')
    ]

    if args.allow_stub:
        cmd.append('--allow-stub')

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=args.project_dir)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description='RATFA Complete Pipeline')
    parser.add_argument('--project-dir', default=os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument('--annotations', required=True, help='COCO annotations JSON')
    parser.add_argument('--images-dir', required=True, help='Images directory')
    parser.add_argument('--output-dir', default='outputs/ratfa')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train-only', action='store_true')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--allow-stub', action='store_true',
                       help='Allow using stub detector (for testing)')
    args = parser.parse_args()

    args.project_dir = os.path.abspath(args.project_dir)
    args.output_dir = os.path.abspath(args.output_dir)

    os.chdir(args.project_dir)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    success = True

    if not args.eval_only:
        success = run_training(args) and success

    if not args.train_only:
        success = run_evaluation(args) and success

    print("\n" + "=" * 60)
    if success:
        print("✅ RATFA PIPELINE COMPLETED SUCCESSFULLY")
    else:
        print("❌ RATFA PIPELINE FAILED")
    print("=" * 60)

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())