"""
Temporal Tracking Module for RATFA

Implements temporal stability estimation τ_t using IoU-based frame association.
This module provides the temporal feedback signal for the RATFA controller.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Track:
    """Represents a tracked object across frames."""
    track_id: int
    boxes: List[np.ndarray] = field(default_factory=list)
    classes: List[int] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    frame_ids: List[int] = field(default_factory=list)


class TemporalModule:
    """
    Temporal stability module for RATFA.
    Computes τ_t (temporal stability) from frame-to-frame associations.
    """

    def __init__(self, iou_threshold: float = 0.3):
        """
        Initialize temporal module.

        Args:
            iou_threshold: IoU threshold for positive association
        """
        self.iou_threshold = iou_threshold
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 0

    def associate_detections(self, boxes: np.ndarray, classes: np.ndarray,
                            scores: np.ndarray) -> Dict[int, int]:
        """
        Associate detections with existing tracks using IoU matching.

        Returns:
            Dictionary mapping detection index -> track_id
        """
        if len(boxes) == 0:
            return {}

        associations = {}
        unmatched = list(range(len(boxes)))

        for track_id, track in self.tracks.items():
            if not unmatched or len(track.boxes) == 0:
                continue

            last_box = track.boxes[-1]
            best_iou = 0
            best_det = -1

            for det_idx in unmatched:
                iou = self._box_iou(boxes[det_idx], last_box)
                if iou > best_iou and iou >= self.iou_threshold:
                    best_iou = iou
                    best_det = det_idx

            if best_det >= 0:
                associations[best_det] = track_id
                unmatched.remove(best_det)

        return associations

    def update_tracks(self, boxes: np.ndarray, classes: np.ndarray,
                     scores: np.ndarray, frame_id: int) -> List[int]:
        """
        Update tracks with new detections.

        Returns:
            List of track IDs for current detections
        """
        associations = self.associate_detections(boxes, classes, scores)

        # Update existing tracks
        for det_idx, track_id in associations.items():
            track = self.tracks[track_id]
            track.boxes.append(boxes[det_idx].copy())
            track.classes.append(classes[det_idx])
            track.scores.append(scores[det_idx])
            track.frame_ids.append(frame_id)

        # Create new tracks for unmatched detections
        associated_dets = set(associations.keys())
        for det_idx in range(len(boxes)):
            if det_idx not in associated_dets:
                new_track = Track(
                    track_id=self.next_track_id,
                    boxes=[boxes[det_idx].copy()],
                    classes=[classes[det_idx]],
                    scores=[scores[det_idx]],
                    frame_ids=[frame_id]
                )
                self.tracks[self.next_track_id] = new_track
                self.next_track_id += 1

        # Return track IDs
        track_ids = []
        for det_idx in range(len(boxes)):
            if det_idx in associations:
                track_ids.append(associations[det_idx])
            else:
                # Find newly created track
                found = False
                for tid, track in self.tracks.items():
                    if len(track.boxes) == 1:
                        if np.allclose(track.boxes[0], boxes[det_idx]):
                            track_ids.append(tid)
                            found = True
                            break
                if not found:
                    track_ids.append(-1)

        return track_ids

    def compute_stability(self) -> float:
        """
        Compute temporal stability τ_t.
        Returns 0.0-1.0 where higher values indicate more consistent tracks.
        """
        if len(self.tracks) == 0:
            return 0.5  # Neutral fallback

        # Count tracks with valid history (multiple frames)
        valid_tracks = sum(1 for t in self.tracks.values() if len(t.boxes) >= 2)

        # Stability is fraction of tracks with history
        return min(1.0, valid_tracks / max(1, len(self.tracks)))

    def _box_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Compute IoU between two boxes [x1, y1, x2, y2]"""
        x1_i = max(float(box1[0]), float(box2[0]))
        y1_i = max(float(box1[1]), float(box2[1]))
        x2_i = min(float(box1[2]), float(box2[2]))
        y2_i = min(float(box1[3]), float(box2[3]))

        inter_w = max(0, x2_i - x1_i)
        inter_h = max(0, y2_i - y1_i)
        inter_area = inter_w * inter_h

        box1_area = (float(box1[2]) - float(box1[0])) * (float(box1[3]) - float(box1[1]))
        box2_area = (float(box2[2]) - float(box2[0])) * (float(box2[3]) - float(box2[1]))

        union_area = box1_area + box2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def reset(self) -> None:
        """Reset all tracks (for new video sequence)."""
        self.tracks = {}
        self.next_track_id = 0

    def get_track_count(self) -> int:
        """Get number of active tracks."""
        return len(self.tracks)

    def get_class_history(self, track_id: int) -> List[int]:
        """Get class history for a track (for class consistency computation)."""
        if track_id in self.tracks:
            return self.tracks[track_id].classes
        return []