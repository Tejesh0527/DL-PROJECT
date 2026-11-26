"""
tracking/temporal_consistency.py
================================
Baseline temporal-consistency estimator (tau_t) and per-track class history.

What it does
------------
Detections of frame t are greedily matched to the tracks of previous frames by
box overlap (IoU).  Then:

  tau_t = mean over current detections of (IoU with its matched track, 0 if unmatched)

so tau_t is close to 1 when objects move smoothly between frames and close to 0
when boxes appear/disappear/jump.  With no previous tracks or no current
detections there is no evidence, so a neutral value (default 0.5) is returned
and `valid` is False.

It also keeps {track_id: [class_id, ...]} so that the class-consistency term of
the reliability descriptor (phi in Eq. 5) can be computed from real history
instead of the neutral 0.5 fallback.

CAVEAT
------
This is a simple IoU-association baseline, not a full multi-object tracker
(no motion model, no appearance features).  It is strictly causal: only the
current and past frames are used.
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Sequence

from detector.fyolo_detector import Detection
from detector.vehicle_detection import compute_iou

logger = logging.getLogger(__name__)


@dataclass
class TrackerOutput:
    """Result of one tracker update."""
    tau: float                                   # temporal consistency in [0, 1]
    valid: bool                                  # False -> neutral fallback used
    class_history: Dict[int, List[int]] = field(default_factory=dict)
    track_ids: List[int] = field(default_factory=list)   # one per input detection
    num_matched: int = 0


class TemporalConsistencyTracker:
    """Greedy IoU tracker producing tau_t and per-track class history."""

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_age: int = 5,
        history_len: int = 10,
        neutral: float = 0.5,
    ):
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(f"iou_threshold must be in [0, 1]; got {iou_threshold}")
        if max_age < 0 or history_len < 1:
            raise ValueError("max_age must be >= 0 and history_len >= 1")
        if not 0.0 <= neutral <= 1.0:
            raise ValueError(f"neutral must be in [0, 1]; got {neutral}")
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.history_len = history_len
        self.neutral = neutral
        self.reset()

    def reset(self) -> None:
        """Forget all tracks (call at the start of every new video)."""
        self._tracks: Dict[int, Dict] = {}
        self._next_id = 0

    def update(self, detections: Sequence[Detection]) -> TrackerOutput:
        had_tracks = bool(self._tracks)
        track_ids = list(self._tracks.keys())

        # All candidate (iou, det_index, track_id) pairs above the threshold.
        pairs = []
        for di, det in enumerate(detections):
            for tid in track_ids:
                iou = compute_iou(det.bbox, self._tracks[tid]["bbox"])
                if iou >= self.iou_threshold:
                    pairs.append((iou, di, tid))
        pairs.sort(reverse=True)

        det_to_track: Dict[int, int] = {}
        det_iou: Dict[int, float] = {}
        used_tracks = set()
        for iou, di, tid in pairs:
            if di in det_to_track or tid in used_tracks:
                continue
            det_to_track[di] = tid
            det_iou[di] = iou
            used_tracks.add(tid)

        # Update matched tracks, create new ones for unmatched detections.
        assigned: List[int] = []
        for di, det in enumerate(detections):
            if di in det_to_track:
                tid = det_to_track[di]
            else:
                tid = self._next_id
                self._next_id += 1
                self._tracks[tid] = {"bbox": det.bbox, "age": 0, "classes": []}
            tr = self._tracks[tid]
            tr["bbox"] = det.bbox
            tr["age"] = 0
            tr["classes"].append(det.class_id)
            tr["classes"] = tr["classes"][-self.history_len:]
            assigned.append(tid)

        # Age and drop tracks that were not seen this frame.
        for tid in list(self._tracks.keys()):
            if tid not in assigned:
                self._tracks[tid]["age"] += 1
                if self._tracks[tid]["age"] > self.max_age:
                    del self._tracks[tid]

        history = {tid: list(tr["classes"]) for tid, tr in self._tracks.items() if tr["classes"]}

        if not had_tracks or len(detections) == 0:
            return TrackerOutput(
                tau=self.neutral, valid=False, class_history=history,
                track_ids=assigned, num_matched=0,
            )

        tau = sum(det_iou.get(i, 0.0) for i in range(len(detections))) / len(detections)
        return TrackerOutput(
            tau=float(min(max(tau, 0.0), 1.0)), valid=True, class_history=history,
            track_ids=assigned, num_matched=len(det_to_track),
        )
