"""
video_processor.py
===================
Sequential video frame processing for RATFA.

Implements the video processing pipeline:
  1. Load video with OpenCV
  2. Read frames in chronological order
  3. Record frame numbers and timestamps
  4. Run vehicle detection on each frame
  5. Estimate scene conditions
  6. Apply causal temporal smoothing
  7. Draw and save annotated output video
  8. Save per-frame detection JSON
  9. Handle errors gracefully and release resources
"""

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np

from detector.fyolo_detector import BaseDetector, Detection
from detector.vehicle_detection import detect_vehicles, draw_detections, compute_reliability
from scene_analysis.condition_estimator import SceneConditionEstimator, SceneConditionResult
from scene_analysis.condition_smoother import ConditionSmoother
from tracking.temporal_consistency import TemporalConsistencyTracker
from fuzzy.parameter_manager import ControlSignal, FuzzyParameterManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class VideoInfo:
    """Metadata extracted from an OpenCV VideoCapture."""
    path: str
    fps: float
    frame_count: int
    width: int
    height: int
    codec: str

    @property
    def duration_s(self) -> float:
        return self.frame_count / self.fps if self.fps > 0 else 0.0


@dataclass
class FrameResult:
    """Results for a single processed frame."""
    frame_number: int
    timestamp_ms: float
    detections: List[Detection] = field(default_factory=list)
    scene_condition: Optional[Dict] = None
    smoothed_condition: Optional[Dict] = None
    reliability: Optional[Dict] = None
    temporal_consistency: Optional[Dict] = None   # {'tau', 'valid', 'num_matched'}
    adaptation: Optional[Dict] = None             # parameter-update record (if enabled)


# ---------------------------------------------------------------------------
# Video info helper
# ---------------------------------------------------------------------------


def get_video_info(video_path: str) -> VideoInfo:
    """Extract metadata from a video file.

    Raises
    ------
    FileNotFoundError
        If the video path does not exist.
    IOError
        If OpenCV cannot open the file.
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join(chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4))
    cap.release()

    return VideoInfo(
        path=video_path,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        codec=codec,
    )


# ---------------------------------------------------------------------------
# Frame generator
# ---------------------------------------------------------------------------


def frame_generator(
    video_path: str,
    max_frames: Optional[int] = None,
) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield (frame_number, timestamp_ms, frame) tuples in chronological order.

    Parameters
    ----------
    video_path : str
        Path to the video file.
    max_frames : int, optional
        Stop after this many frames (useful for testing).

    Yields
    ------
    (frame_number, timestamp_ms, frame) : (int, float, np.ndarray)
    """
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: {video_path}")

    try:
        frame_number = 0
        while True:
            if max_frames is not None and frame_number >= max_frames:
                break
            ret, frame = cap.read()
            if not ret:
                break
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            yield frame_number, timestamp_ms, frame
            frame_number += 1
    finally:
        cap.release()


# ---------------------------------------------------------------------------
# Video processor
# ---------------------------------------------------------------------------


class VideoProcessor:
    """Full RATFA video processing pipeline.

    Parameters
    ----------
    detector : BaseDetector
        Loaded detector instance.
    scene_estimator : SceneConditionEstimator
        Scene-condition estimator.
    condition_smoother : ConditionSmoother
        Causal temporal smoother (reset at the start of each video).
    display : bool
        If True, show frames in a window (requires a display).
    """

    def __init__(
        self,
        detector: BaseDetector,
        scene_estimator: SceneConditionEstimator,
        condition_smoother: ConditionSmoother,
        display: bool = False,
        tracker: Optional[TemporalConsistencyTracker] = None,
        parameter_manager: Optional[FuzzyParameterManager] = None,
    ):
        self.detector = detector
        self.scene_estimator = scene_estimator
        self.condition_smoother = condition_smoother
        self.display = display
        # Optional extras.  With both left as None the pipeline behaves exactly
        # as before (no tau_t, neutral class consistency, no adaptation).
        self.tracker = tracker
        self.parameter_manager = parameter_manager
        self._warned_no_apply_hook = False

    def process_video(
        self,
        video_path: str,
        output_video_path: Optional[str] = None,
        output_json_path: Optional[str] = None,
        output_scene_path: Optional[str] = None,
        output_codec: str = "mp4v",
        max_frames: Optional[int] = None,
    ) -> List[FrameResult]:
        """Process a video file frame by frame.

        Parameters
        ----------
        video_path : str
            Input video.
        output_video_path : str, optional
            Destination for the annotated output video.
        output_json_path : str, optional
            Destination for the per-frame detection JSON.
        output_scene_path : str, optional
            Destination for the per-frame scene-condition JSON.
        output_codec : str
            FourCC codec string for the output video.
        max_frames : int, optional
            Process only this many frames (useful for testing).

        Returns
        -------
        List[FrameResult]
            One entry per frame.
        """
        # --- Validate input ---
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Input video not found: {video_path}")

        info = get_video_info(video_path)
        logger.info(
            "Processing video: %s  (%.1f fps, %d frames, %dx%d)",
            video_path, info.fps, info.frame_count, info.width, info.height,
        )

        # --- Reset smoother for new video ---
        self.condition_smoother.reset()
        if self.tracker is not None:
            self.tracker.reset()
        if self.parameter_manager is not None:
            self.parameter_manager.reset()

        # --- Prepare output writer ---
        writer: Optional[cv2.VideoWriter] = None
        if output_video_path:
            os.makedirs(
                os.path.dirname(os.path.abspath(output_video_path)), exist_ok=True
            )
            fourcc = cv2.VideoWriter_fourcc(*output_codec)
            writer = cv2.VideoWriter(
                output_video_path,
                fourcc,
                info.fps,
                (info.width, info.height),
            )
            if not writer.isOpened():
                logger.warning(
                    "Could not open VideoWriter for %s; output video will not be saved.",
                    output_video_path,
                )
                writer = None

        frame_results: List[FrameResult] = []

        try:
            for frame_number, timestamp_ms, frame in frame_generator(
                video_path, max_frames=max_frames
            ):
                result = self._process_frame(frame, frame_number, timestamp_ms)
                frame_results.append(result)

                # Annotate and write
                annotated = draw_detections(frame, result.detections)
                self._overlay_scene_info(annotated, result)

                if writer is not None:
                    writer.write(annotated)

                if self.display:
                    cv2.imshow("RATFA", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        logger.info("User interrupted video processing.")
                        break

                if frame_number % 50 == 0:
                    logger.info("Processed frame %d / %d", frame_number, info.frame_count)

        finally:
            if writer is not None:
                writer.release()
            if self.display:
                cv2.destroyAllWindows()

        # --- Save outputs ---
        if output_json_path:
            self._save_detections_json(frame_results, output_json_path, info)

        if output_scene_path:
            self._save_scene_json(frame_results, output_scene_path, info)

        logger.info("Video processing complete: %d frames.", len(frame_results))
        return frame_results

    # ------------------------------------------------------------------
    # Per-frame processing
    # ------------------------------------------------------------------

    def _process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp_ms: float,
    ) -> FrameResult:
        """Run detection and scene analysis on a single frame."""
        # Detection
        detections = detect_vehicles(
            self.detector, frame, frame_number=frame_number, timestamp_ms=timestamp_ms
        )

        # Scene estimation
        try:
            scene_result = self.scene_estimator.estimate(frame)
            scene_dict = scene_result.to_dict()
        except Exception as exc:
            logger.warning("Scene estimation failed for frame %d: %s", frame_number, exc)
            scene_dict = {"clear": 0.25, "rain": 0.25, "fog": 0.25, "dark": 0.25,
                          "scene_confidence": 0.0}
            scene_result = None

        # Causal smoothing
        if scene_result is not None:
            smoothed_vec = self.condition_smoother.update(scene_result.vector)
        else:
            import numpy as np
            smoothed_vec = np.ones(4) / 4.0

        smoothed_dict = {
            "clear": float(smoothed_vec[0]),
            "rain": float(smoothed_vec[1]),
            "fog": float(smoothed_vec[2]),
            "dark": float(smoothed_vec[3]),
        }

        # Temporal consistency tau_t and class history (optional)
        tc_dict = None
        class_history = None
        tau = None
        if self.tracker is not None:
            tr_out = self.tracker.update(detections)
            tau = tr_out.tau
            class_history = tr_out.class_history
            tc_dict = {"tau": tr_out.tau, "valid": tr_out.valid,
                       "num_matched": tr_out.num_matched}

        # Detection reliability (Eqs. 3-5)
        reliability = compute_reliability(detections, class_history=class_history)

        # RATFA adaptation step (Eqs. 7-11), computed on frame t and meant for t+1
        adaptation = None
        if self.parameter_manager is not None:
            adaptation = self._adapt(smoothed_vec, scene_dict, reliability, tau, frame_number)

        return FrameResult(
            frame_number=frame_number,
            timestamp_ms=timestamp_ms,
            detections=detections,
            scene_condition=scene_dict,
            smoothed_condition=smoothed_dict,
            reliability=reliability,
            temporal_consistency=tc_dict,
            adaptation=adaptation,
        )

    def _adapt(self, smoothed_vec, scene_dict, reliability, tau, frame_number) -> Dict:
        """Run one gated, bounded parameter update and return its record."""
        if tau is None or not reliability.get("valid", False):
            # No detections (or no tracker) -> no evidence -> do not adapt.
            return {"applied": False, "reason": "insufficient evidence (no valid reliability / tau)"}
        signal = ControlSignal(
            smoothed_condition=np.asarray(smoothed_vec, dtype=np.float64),
            reliability=float(reliability["reliability"]),
            temporal_consistency=float(tau),
            scene_confidence=float(scene_dict.get("scene_confidence", 0.0)),
        )
        new_params = self.parameter_manager.update(signal, frame_number=frame_number)
        record = self.parameter_manager.get_update_log()[-1]
        apply_hook = getattr(self.detector, "apply_fuzzy_parameters", None)
        if callable(apply_hook):
            apply_hook(new_params)           # takes effect from the next frame
        elif not self._warned_no_apply_hook:
            logger.warning(
                "Detector '%s' has no apply_fuzzy_parameters(); adaptation is "
                "computed and logged but NOT applied to any model.",
                getattr(self.detector, "backend_name", type(self.detector).__name__),
            )
            self._warned_no_apply_hook = True
        record["applied"] = bool(record.get("update_applied", True))
        return record

    # ------------------------------------------------------------------
    # Overlay
    # ------------------------------------------------------------------

    @staticmethod
    def _overlay_scene_info(frame: np.ndarray, result: FrameResult) -> None:
        """Draw scene-condition text overlay on the annotated frame."""
        if result.smoothed_condition is None:
            return
        sc = result.smoothed_condition
        dominant = max(sc, key=sc.get)
        rel_str = f" | Rel: {result.reliability['reliability']:.2f}" if result.reliability else ""
        tau_str = (f" | Tau: {result.temporal_consistency['tau']:.2f}"
                   if result.temporal_consistency else "")
        text = (
            f"Frame {result.frame_number} | "
            f"Scene: {dominant} | "
            f"Dets: {len(result.detections)}{rel_str}{tau_str}"
        )
        cv2.putText(
            frame, text, (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA,
        )

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------

    def _save_detections_json(
        self,
        frame_results: List[FrameResult],
        path: str,
        info: VideoInfo,
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        payload = {
            "source_video": info.path,
            "fps": info.fps,
            "total_frames": len(frame_results),
            "frames": [
                {
                    "frame_number": r.frame_number,
                    "timestamp_ms": r.timestamp_ms,
                    "num_detections": len(r.detections),
                    "detections": [d.to_dict() for d in r.detections],
                }
                for r in frame_results
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Detections JSON saved → %s", path)

    def _save_scene_json(
        self,
        frame_results: List[FrameResult],
        path: str,
        info: VideoInfo,
    ) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        def _frame(r: FrameResult) -> Dict:
            d = {
                "frame_number": r.frame_number,
                "timestamp_ms": r.timestamp_ms,
                "raw_condition": r.scene_condition,
                "smoothed_condition": r.smoothed_condition,
                "reliability": r.reliability,
            }
            if r.temporal_consistency is not None:
                d["temporal_consistency"] = r.temporal_consistency
            if r.adaptation is not None:
                d["adaptation"] = r.adaptation
            return d

        payload = {
            "source_video": info.path,
            "frames": [_frame(r) for r in frame_results],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Scene analysis JSON saved → %s", path)

