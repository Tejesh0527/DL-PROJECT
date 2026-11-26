"""
main.py
========
RATFA – Main Application Entry Point

Reliability-Aware Temporal Fuzzy Adaptation for Robust Traffic Object
Detection and Vehicle Tracking.

Usage
-----
Image mode:
  python main.py --mode image --input data/input_images/test.jpg

Video mode:
  python main.py --mode video --input data/input_videos/test.mp4

Options:
  --config    Path to config.yaml   (default: config/config.yaml)
  --output    Output directory      (overrides config defaults)
  --no-display                      Suppress frame display window
  --max-frames N                    Process only first N frames (video mode)
  --save-params                     Save fuzzy parameter inspection report
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Optional

# ---------------------------------------------------------------------------
# Path setup – allow running from project root without install
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from utils.config_loader import load_config
from utils.logger_setup import setup_logging
from detector.fyolo_detector import create_detector
from detector.vehicle_detection import (
    detect_vehicles,
    draw_detections,
    save_annotated_image,
    save_detections_json,
    compute_reliability,
)
from scene_analysis.condition_estimator import SceneConditionEstimator
from scene_analysis.condition_smoother import ConditionSmoother
from fuzzy.membership import gaussian_membership
from fuzzy.parameter_inspector import FuzzyParameterInspector
# Runtime parameter adaptation (--adapt) runs the gated, bounded update every
# frame in video mode, but with MOCK fuzzy parameters: real F-YOLO layers are
# unavailable, so updates are logged and not applied to any actual model.
from video_processor import VideoProcessor, get_video_info

logger = logging.getLogger("ratfa.main")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="RATFA: Reliability-Aware Temporal Fuzzy Adaptation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--mode", choices=["image", "video"], required=True,
        help="Processing mode: 'image' or 'video'.",
    )
    p.add_argument(
        "--input", required=True,
        help="Path to input image or video file.",
    )
    p.add_argument(
        "--config", default=os.path.join("config", "config.yaml"),
        help="Path to config.yaml  (default: config/config.yaml).",
    )
    p.add_argument(
        "--output", default=None,
        help="Output directory (overrides config defaults).",
    )
    p.add_argument(
        "--no-display", action="store_true",
        help="Suppress display window.",
    )
    p.add_argument(
        "--max-frames", type=int, default=None,
        help="Process only the first N frames (video mode).",
    )
    p.add_argument(
        "--adapt", action="store_true",
        help="Video mode: compute tau_t and run the gated RATFA parameter update "
             "each frame (uses MOCK fuzzy parameters while F-YOLO is unavailable).",
    )
    p.add_argument(
        "--save-params", action="store_true",
        help="Inspect and save the fuzzy parameter report.",
    )
    return p


# ---------------------------------------------------------------------------
# Shared initialisation
# ---------------------------------------------------------------------------


def initialise(config_path: str):
    """Load config, logging, detector, estimators, and smoother."""
    cfg = load_config(config_path)

    log_cfg = cfg.get("logging", {})
    setup_logging(
        level=log_cfg.get("level", "INFO"),
        log_dir=log_cfg.get("log_dir", "logs"),
        log_file=log_cfg.get("log_file", "ratfa.log"),
        fmt=log_cfg.get(
            "format",
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        ),
    )
    logger.info("RATFA initialising – %s", datetime.now().isoformat())

    # Detector
    det_cfg = cfg.get("detector", {})
    detector = create_detector(det_cfg)

    # Scene estimator
    scene_cfg = cfg.get("scene", {})
    heuristic_params = scene_cfg.get("heuristic", {})
    estimator = SceneConditionEstimator(
        method=scene_cfg.get("method", "heuristic"),
        heuristic_params=heuristic_params,
    )

    # Smoother
    smoother_cfg = cfg.get("smoother", {})
    smoother = ConditionSmoother(
        lambda_e=smoother_cfg.get("lambda_e", 0.7),
        num_conditions=smoother_cfg.get("num_conditions", 4),
    )

    return cfg, detector, estimator, smoother


# ---------------------------------------------------------------------------
# Fuzzy parameter inspection (optional)
# ---------------------------------------------------------------------------


def run_parameter_inspection(
    model,   # None when F-YOLO BLOCKED
    output_dir: str,
) -> None:
    """Inspect fuzzy parameters and save report."""
    inspector = FuzzyParameterInspector(model=model)
    report = inspector.inspect()
    issues = inspector.validate_parameters(report)

    report_path = os.path.join(output_dir, "parameter_report.json")
    os.makedirs(output_dir, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    logger.info("Fuzzy parameter report saved → %s", report_path)
    logger.info("Parameter summary:\n%s", report.summary())

    if issues:
        for issue in issues:
            logger.warning("Parameter validation issue: %s", issue)
    else:
        logger.info("All parameter validation checks passed.")


# ---------------------------------------------------------------------------
# Image mode
# ---------------------------------------------------------------------------


def run_image_mode(
    input_path: str,
    cfg: dict,
    detector,
    estimator: SceneConditionEstimator,
    smoother: ConditionSmoother,
    output_dir: Optional[str],
) -> None:
    """Process a single image."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input image not found: {input_path}")

    frame = cv2.imread(input_path)
    if frame is None:
        raise IOError(f"OpenCV could not read image: {input_path}")

    logger.info("Image mode – loaded %s (%dx%d)", input_path, frame.shape[1], frame.shape[0])

    # --- Detection ---
    detections = detect_vehicles(detector, frame, frame_number=0)
    logger.info("Detected %d vehicle(s).", len(detections))

    # --- Scene condition ---
    scene_result = estimator.estimate(frame)
    logger.info("Scene: %s", scene_result)

    # --- Scene confidence ---
    q_t = scene_result.scene_confidence
    logger.info("Scene confidence q_t = %.4f", q_t)

    # --- Causal smoothing (single frame: smoother initialises to current) ---
    smoother.reset()
    smoothed_vec = smoother.update(scene_result.vector)
    smoothed_dict = {
        "clear": float(smoothed_vec[0]),
        "rain": float(smoothed_vec[1]),
        "fog": float(smoothed_vec[2]),
        "dark": float(smoothed_vec[3]),
    }
    logger.info("Smoothed scene vector: %s", smoothed_dict)

    # --- Reliability ---
    reliability = compute_reliability(detections)
    logger.info("Reliability: %.4f", reliability["reliability"])

    # --- Determine output paths ---
    base = os.path.splitext(os.path.basename(input_path))[0]
    outputs_cfg = cfg.get("outputs", {})
    det_dir = output_dir or outputs_cfg.get("detections_dir", "outputs/detections")
    scene_dir = output_dir or outputs_cfg.get("scene_analysis_dir", "outputs/scene_analysis")

    annotated_path = os.path.join(det_dir, f"{base}_annotated.jpg")
    json_path = os.path.join(det_dir, f"{base}_detections.json")
    scene_path = os.path.join(scene_dir, f"{base}_scene.json")

    # --- Save outputs ---
    annotated = draw_detections(frame, detections)
    save_annotated_image(annotated, annotated_path)

    save_detections_json(
        detections,
        json_path,
        metadata={
            "input": input_path,
            "detector": detector.backend_name,
            "mode": "image",
        },
    )

    os.makedirs(os.path.dirname(os.path.abspath(scene_path)), exist_ok=True)
    scene_report = {
        "input": input_path,
        "frame_number": 0,
        "raw_condition": scene_result.to_dict(),
        "smoothed_condition": smoothed_dict,
        "reliability": reliability,
        "raw_features": scene_result.raw_features,
    }
    with open(scene_path, "w", encoding="utf-8") as f:
        json.dump(scene_report, f, indent=2)
    logger.info("Scene report saved → %s", scene_path)

    logger.info(
        "Image mode complete.\n"
        "  Annotated image : %s\n"
        "  Detections JSON : %s\n"
        "  Scene report    : %s",
        annotated_path, json_path, scene_path,
    )


# ---------------------------------------------------------------------------
# Video mode
# ---------------------------------------------------------------------------


def run_video_mode(
    input_path: str,
    cfg: dict,
    detector,
    estimator: SceneConditionEstimator,
    smoother: ConditionSmoother,
    output_dir: Optional[str],
    display: bool = False,
    max_frames: Optional[int] = None,
    adapt: bool = False,
) -> None:
    """Process a video file frame by frame."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    base = os.path.splitext(os.path.basename(input_path))[0]
    outputs_cfg = cfg.get("outputs", {})
    vid_dir = output_dir or outputs_cfg.get("videos_dir", "outputs/videos")
    det_dir = output_dir or outputs_cfg.get("detections_dir", "outputs/detections")
    scene_dir = output_dir or outputs_cfg.get("scene_analysis_dir", "outputs/scene_analysis")

    out_vid = os.path.join(vid_dir, f"{base}_ratfa.mp4")
    out_json = os.path.join(det_dir, f"{base}_detections.json")
    out_scene = os.path.join(scene_dir, f"{base}_scene.json")

    video_cfg = cfg.get("video", {})
    codec = video_cfg.get("output_codec", "mp4v")

    tracker = None
    manager = None
    adapt = adapt or bool(cfg.get("adaptation", {}).get("enabled", False))
    if adapt:
        from tracking.temporal_consistency import TemporalConsistencyTracker
        from fuzzy.parameter_inspector import FuzzyParameterInspector
        from fuzzy.parameter_manager import FuzzyParameterManager
        a_cfg = cfg.get("adaptation", {})
        pm_cfg = cfg.get("parameter_manager", {})
        tracker = TemporalConsistencyTracker(
            iou_threshold=a_cfg.get("tracker_iou_threshold", 0.3),
            max_age=a_cfg.get("tracker_max_age", 5),
        )
        # NOTE: while F-YOLO is unavailable the inspector returns a MOCK report.
        reference = FuzzyParameterInspector(model=None).inspect()
        manager = FuzzyParameterManager(
            reference_report=reference,
            max_centre_frac=pm_cfg.get("max_center_fractional_change", 0.10),
            max_sigma_frac=pm_cfg.get("max_width_fractional_change", 0.15),
            gate_threshold=a_cfg.get("gate_threshold", 0.1),
            rate_limit=a_cfg.get("rate_limit", 0.05),
        )
        logger.warning(
            "Adaptation ENABLED with MOCK fuzzy parameters (F-YOLO unavailable): "
            "updates are logged but do not change any real model."
        )

    processor = VideoProcessor(
        detector=detector,
        scene_estimator=estimator,
        condition_smoother=smoother,
        display=display,
        tracker=tracker,
        parameter_manager=manager,
    )

    frame_results = processor.process_video(
        video_path=input_path,
        output_video_path=out_vid,
        output_json_path=out_json,
        output_scene_path=out_scene,
        output_codec=codec,
        max_frames=max_frames,
    )

    if manager is not None:
        rep_dir = output_dir or outputs_cfg.get("parameter_reports_dir", "outputs/parameter_reports")
        manager.save_update_log(os.path.join(rep_dir, f"{base}_adaptation_log.json"))
        manager.save_current_parameters(os.path.join(rep_dir, f"{base}_final_parameters_MOCK.json"))

    logger.info(
        "Video mode complete: %d frames processed.\n"
        "  Annotated video : %s\n"
        "  Detections JSON : %s\n"
        "  Scene analysis  : %s",
        len(frame_results), out_vid, out_json, out_scene,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        cfg, detector, estimator, smoother = initialise(args.config)
    except Exception as exc:
        print(f"[FATAL] Initialisation failed: {exc}", file=sys.stderr)
        return 1

    # Optional fuzzy parameter inspection
    if args.save_params:
        param_dir = cfg.get("outputs", {}).get(
            "parameter_reports_dir", "outputs/parameter_reports"
        )
        if args.output:
            param_dir = args.output
        try:
            # Pass None as model because F-YOLO is BLOCKED
            run_parameter_inspection(model=None, output_dir=param_dir)
        except Exception as exc:
            logger.error("Parameter inspection failed: %s", exc)

    try:
        if args.mode == "image":
            run_image_mode(
                input_path=args.input,
                cfg=cfg,
                detector=detector,
                estimator=estimator,
                smoother=smoother,
                output_dir=args.output,
            )
        elif args.mode == "video":
            run_video_mode(
                input_path=args.input,
                cfg=cfg,
                detector=detector,
                estimator=estimator,
                smoother=smoother,
                output_dir=args.output,
                display=not args.no_display,
                max_frames=args.max_frames,
                adapt=args.adapt,
            )
    except FileNotFoundError as exc:
        logger.error("Input not found: %s", exc)
        return 2
    except IOError as exc:
        logger.error("IO error: %s", exc)
        return 3
    except Exception as exc:
        logger.exception("Unexpected error: %s", exc)
        return 4

    return 0


if __name__ == "__main__":
    sys.exit(main())

