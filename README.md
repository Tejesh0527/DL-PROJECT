# RATFA

## Reliability-Aware Temporal Fuzzy Adaptation for Robust Traffic Object Detection and Vehicle Tracking

A computer vision research project developed at SRM University AP.

## Objective

Improve the reliability of traffic object detection and vehicle tracking under changing environmental conditions using fuzzy logic and temporal adaptation.

## Project Components

- YOLO-based vehicle detection
- Environmental scene estimation
- Gaussian fuzzy membership functions
- Detection reliability estimation
- Temporal consistency estimation
- Confidence-gated parameter adaptation
- Streamlit dashboard

## Technologies

Python, PyTorch, YOLOv8, OpenCV, NumPy, SciPy and Streamlit.

## Installation

    python -m pip install -r requirements.txt

## Run Dashboard

    python -m streamlit run app.py

## Run Tests

    python -m pytest tests/ -v

## Research Status

The original F-YOLO weights are unavailable.
YOLOv8 is used as an alternative implementation.
Real-world detection, tracking and adaptation performance require further experimental verification.
Mock outputs are intended only for software testing.

## Institution

SRM University AP
Department of Computer Science and Engineering
