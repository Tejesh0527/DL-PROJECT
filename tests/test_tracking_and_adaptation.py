"""Tests for the temporal-consistency tracker and the adaptation wiring."""
import numpy as np
import pytest

from detector.fyolo_detector import Detection, StubDetector
from tracking.temporal_consistency import TemporalConsistencyTracker
from scene_analysis.condition_estimator import SceneConditionEstimator
from scene_analysis.condition_smoother import ConditionSmoother
from fuzzy.parameter_inspector import FuzzyParameterInspector
from fuzzy.parameter_manager import FuzzyParameterManager
from video_processor import VideoProcessor
import cv2


def det(box, cls=2, conf=0.9, f=0):
    return Detection(frame_number=f, bbox=box, class_id=cls, class_name="car", confidence=conf)


class TestTracker:
    def test_first_frame_is_neutral_and_invalid(self):
        t = TemporalConsistencyTracker()
        out = t.update([det((10, 10, 50, 50))])
        assert out.tau == 0.5 and out.valid is False

    def test_static_object_gives_tau_one(self):
        t = TemporalConsistencyTracker()
        t.update([det((10, 10, 50, 50))])
        out = t.update([det((10, 10, 50, 50))])
        assert out.valid and out.tau == pytest.approx(1.0)
        assert out.num_matched == 1

    def test_jumping_object_gives_tau_zero(self):
        t = TemporalConsistencyTracker()
        t.update([det((0, 0, 20, 20))])
        out = t.update([det((300, 300, 340, 340))])
        assert out.valid and out.tau == 0.0

    def test_tau_always_in_unit_interval(self):
        t = TemporalConsistencyTracker()
        rng = np.random.default_rng(0)
        for _ in range(30):
            x = rng.uniform(0, 80)
            out = t.update([det((x, x, x + 30, x + 30))])
            assert 0.0 <= out.tau <= 1.0

    def test_class_history_accumulates_per_track(self):
        t = TemporalConsistencyTracker()
        t.update([det((10, 10, 50, 50), cls=2)])
        out = t.update([det((11, 11, 51, 51), cls=7)])
        assert list(out.class_history.values()) == [[2, 7]]

    def test_no_detections_after_tracks_is_neutral(self):
        t = TemporalConsistencyTracker()
        t.update([det((10, 10, 50, 50))])
        out = t.update([])
        assert out.tau == 0.5 and out.valid is False

    def test_old_tracks_are_dropped(self):
        t = TemporalConsistencyTracker(max_age=1)
        t.update([det((10, 10, 50, 50))])
        t.update([]); t.update([]); t.update([])
        assert t._tracks == {}

    def test_reset_clears_state(self):
        t = TemporalConsistencyTracker()
        t.update([det((10, 10, 50, 50))])
        t.reset()
        assert t.update([det((10, 10, 50, 50))]).valid is False

    def test_invalid_args(self):
        with pytest.raises(ValueError):
            TemporalConsistencyTracker(iou_threshold=2.0)


def _make_video(path, n=8, w=160, h=120):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    for i in range(n):
        wr.write(np.full((h, w, 3), 100 + i, np.uint8))
    wr.release()


class TestVideoAdaptationWiring:
    def _processor(self, **kw):
        det_ = StubDetector(); det_.load_model()
        return VideoProcessor(det_, SceneConditionEstimator(), ConditionSmoother(), **kw)

    def test_default_behaviour_unchanged(self, tmp_path):
        v = tmp_path / "v.mp4"; _make_video(v)
        res = self._processor().process_video(str(v))
        assert all(r.temporal_consistency is None and r.adaptation is None for r in res)

    def test_adaptation_runs_and_is_bounded(self, tmp_path):
        v = tmp_path / "v.mp4"; _make_video(v)
        ref = FuzzyParameterInspector(model=None).inspect()
        mgr = FuzzyParameterManager(ref)
        proc = self._processor(tracker=TemporalConsistencyTracker(), parameter_manager=mgr)
        res = proc.process_video(str(v))
        assert all(r.temporal_consistency is not None and r.adaptation is not None for r in res)
        for cur, refl in zip(mgr._current_params, ref.layers):
            assert np.all(cur.sigmas > 0)
            assert np.all(np.abs(cur.centres - refl.centres) <= np.abs(refl.centres) * 0.10 + 1e-9)
            assert np.all(np.abs(cur.sigmas - refl.sigmas) <= np.abs(refl.sigmas) * 0.15 + 1e-9)

    def test_no_update_without_valid_evidence(self):
        proc = self._processor(tracker=TemporalConsistencyTracker(),
                               parameter_manager=FuzzyParameterManager(FuzzyParameterInspector(model=None).inspect()))
        out = proc._adapt(np.ones(4) / 4, {"scene_confidence": 1.0},
                          {"reliability": 0.5, "valid": False}, 0.5, 0)
        assert out["applied"] is False
