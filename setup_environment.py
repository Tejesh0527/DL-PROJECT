#!/usr/bin/env python3
"""
Environment Setup Script for RATFA
Installs all dependencies and verifies the setup
"""

import subprocess
import sys
import os


def install_packages():
    """Install required packages."""
    packages = [
        'torch',
        'torchvision',
        'opencv-python',
        'numpy',
        'scipy',
        'pyyaml',
        'scikit-fuzzy',
        'pandas',
        'streamlit',
        'plotly',
        'matplotlib',
        'pytest',
        'pytest-cov',
        'ultralytics',
        'filterpy',
    ]

    print("Installing packages...")
    for pkg in packages:
        try:
            subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pkg],
                           check=True, capture_output=True)
            print(f"  ✓ {pkg}")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {pkg}: {e}")


def verify_installation():
    """Verify all key packages can be imported."""
    print("\nVerifying installation...")
    imports = [
        ('torch', 'PyTorch'),
        ('cv2', 'OpenCV'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML'),
        ('skfuzzy', 'scikit-fuzzy'),
        ('pandas', 'Pandas'),
        ('streamlit', 'Streamlit'),
        ('plotly', 'Plotly'),
        ('ultralytics', 'Ultralytics YOLOv8'),
    ]

    for module, name in imports:
        try:
            __import__(module)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")


def download_yolo_weights():
    """Download YOLOv8 weights."""
    print("\nDownloading YOLOv8 weights...")
    try:
        from ultralytics import YOLO
        model = YOLO('yolov8n.pt')  # Download weights
        print("  ✓ YOLOv8 weights downloaded")
    except Exception as e:
        print(f"  ✗ Failed to download weights: {e}")


def main():
    print("=" * 60)
    print("RATFA Environment Setup")
    print("=" * 60)

    install_packages()
    verify_installation()

    response = input("\nDownload YOLOv8 weights? (y/n): ")
    if response.lower() == 'y':
        download_yolo_weights()

    print("\n" + "=" * 60)
    print("Environment setup complete!")
    print("=" * 60)


if __name__ == '__main__':
    main()