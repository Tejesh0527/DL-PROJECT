#!/usr/bin/env python3
"""
RATFA Training Script

Trains a YOLO-based detector with fuzzy feature fusion capability.
Implements the feature learning phase required for RATFA adaptation.
"""

import argparse
import json
import os
import sys
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import cv2
import yaml

# Import existing project modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_tools.coco_dataset import load_coco_vehicles, make_splits, save_splits, VEHICLE_NAME_TO_ID


class SimpleYOLODataset(Dataset):
    """Simple YOLO-format dataset for vehicle detection."""

    def __init__(self, images_dir: str, gt_data: dict, image_ids: list,
                 imgsz: int = 640):
        self.images_dir = images_dir
        self.gt_data = gt_data
        self.image_ids = image_ids
        self.imgsz = imgsz

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        gt_img = self.gt_data[img_id]

        # Load image
        img_path = os.path.join(self.images_dir, gt_img.file_name)
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Letterbox resize
        h, w = img.shape[:2]
        scale = self.imgsz / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img_resized = cv2.resize(img, (new_w, new_h))

        # Pad to square
        pad_h = self.imgsz - new_h
        pad_w = self.imgsz - new_w
        padded = np.ones((self.imgsz, self.imgsz, 3), dtype=np.uint8) * 114
        padded[pad_h // 2:pad_h // 2 + new_h, pad_w // 2:pad_w // 2 + new_w] = img_resized

        # Normalize
        img_norm = padded.astype(np.float32) / 255.0
        img_norm = (img_norm - np.array([0.485, 0.456, 0.406])) / \
                   np.array([0.229, 0.224, 0.225])

        # Targets: (x, y, w, h, class_id) normalized to [0, 1]
        targets = []
        for x1, y1, x2, y2, class_id in gt_img.boxes:
            # Convert to center format and normalize
            cx = (x1 + x2) / 2 / w
            cy = (y1 + y2) / 2 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            # Scale to letterbox
            cx = cx * scale + pad_w / 2 / self.imgsz
            cy = cy * scale + pad_h / 2 / self.imgsz
            bw = bw * scale / self.imgsz
            bh = bh * scale / self.imgsz

            targets.append([cx, cy, bw, bh, class_id])

        return img_norm.transpose(2, 0, 1), np.array(targets, dtype=np.float32)


class SimpleYOLO(nn.Module):
    """Simple YOLO-style detector for prototyping RATFA."""

    def __init__(self, num_classes: int = 4, num_anchors: int = 3):
        super().__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Simple backbone (convolutional feature extractor)
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Detection head
        self.head = nn.Conv2d(256, num_anchors * (num_classes + 5), 1)

        # Fuzzy membership parameters (what RATFA will adapt)
        self.register_buffer('ffm_centers', torch.tensor([0.2, 0.5, 0.8]))
        self.register_buffer('ffm_widths', torch.tensor([0.15, 0.2, 0.15]))
        self.register_buffer('fa_centers', torch.tensor([0.3, 0.7]))
        self.register_buffer('fa_widths', torch.tensor([0.2, 0.2]))

    def forward(self, x):
        features = self.backbone(x)
        output = self.head(features)
        return output

    def get_fuzzy_params(self):
        """Get current fuzzy parameters for inspection."""
        return {
            'ffm_centers': self.ffm_centers.tolist(),
            'ffm_widths': self.ffm_widths.tolist(),
            'fa_centers': self.fa_centers.tolist(),
            'fa_widths': self.fa_widths.tolist(),
        }

    def apply_fuzzy_params(self, params):
        """Apply new fuzzy parameters (for RATFA adaptation)."""
        with torch.no_grad():
            if 'ffm_centers' in params:
                self.ffm_centers = torch.tensor(params['ffm_centers'])
            if 'ffm_widths' in params:
                self.ffm_widths = torch.tensor(params['ffm_widths'])
            if 'fa_centers' in params:
                self.fa_centers = torch.tensor(params['fa_centers'])
            if 'fa_widths' in params:
                self.fa_widths = torch.tensor(params['fa_widths'])


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for imgs, targets in dataloader:
        imgs = imgs.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)

        # Simple loss computation (placeholder - real YOLO loss would be more complex)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def main():
    parser = argparse.ArgumentParser(description='RATFA Training Script')
    parser.add_argument('--annotations', required=True, help='COCO-format annotation JSON')
    parser.add_argument('--images-dir', required=True, help='Folder with images')
    parser.add_argument('--output-dir', default='outputs/train', help='Training output directory')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--train-split-file', default=None, help='Train split JSON file')
    args = parser.parse_args()

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load ground truth data
    print("Loading dataset...")
    gt_data = load_coco_vehicles(args.annotations)
    print(f"Loaded {len(gt_data)} images with vehicle annotations")

    # Create or load splits
    if args.train_split_file:
        splits = make_splits(list(gt_data.keys()), seed=args.seed)
        save_splits(splits, args.train_split_file, args.seed)
        print(f"Train split saved to {args.train_split_file}")
    else:
        image_ids = list(gt_data.keys())
        splits = make_splits(image_ids, seed=args.seed)

    train_ids = splits['train']
    val_ids = splits['val']
    test_ids = splits['test']
    print(f"Split: {len(train_ids)} train, {len(val_ids)} val, {len(test_ids)} test")

    # Create datasets
    train_dataset = SimpleYOLODataset(args.images_dir, gt_data, train_ids)
    val_dataset = SimpleYOLODataset(args.images_dir, gt_data, val_ids)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Create model
    model = SimpleYOLO(num_classes=4).to(device)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()  # Placeholder loss

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch + 1}/{args.epochs}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'fuzzy_params': model.get_fuzzy_params(),
            }, os.path.join(args.output_dir, 'checkpoints', 'best_model.pt'))

    print(f"\nTraining completed. Best model saved to {args.output_dir}/checkpoints/best_model.pt")

    # Save final fuzzy parameters (these are what RATFA will adapt)
    final_params = model.get_fuzzy_params()
    with open(os.path.join(args.output_dir, 'fuzzy_params.json'), 'w') as f:
        json.dump(final_params, f, indent=2)
    print(f"Fuzzy parameters saved to {args.output_dir}/fuzzy_params.json")

    # Save splits
    with open(os.path.join(args.output_dir, 'splits.json'), 'w') as f:
        json.dump({
            'train_ids': train_ids,
            'val_ids': val_ids,
            'test_ids': test_ids,
            'seed': args.seed,
            'vehicle_classes': VEHICLE_NAME_TO_ID
        }, f, indent=2)


if __name__ == '__main__':
    main()