"""Tests for dataset loading, splitting and evaluation metrics (synthetic fixtures only)."""
import json
import pytest

from data_tools.coco_dataset import load_coco_vehicles, make_splits, save_splits, load_splits
from data_tools.evaluation import evaluate_detections, average_precision


@pytest.fixture
def coco_file(tmp_path):
    data = {
        "images": [{"id": i, "file_name": f"{i}.jpg", "width": 100, "height": 100} for i in range(1, 11)],
        "categories": [{"id": 1, "name": "person"}, {"id": 3, "name": "Car"}, {"id": 8, "name": "truck"}],
        "annotations": [
            {"image_id": 1, "category_id": 3, "bbox": [10, 10, 20, 20]},
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 5, 5]},       # person -> ignored
            {"image_id": 2, "category_id": 8, "bbox": [30, 30, 40, 40]},
            {"image_id": 2, "category_id": 3, "bbox": [0, 0, 0, 10]},      # zero width -> ignored
        ],
    }
    p = tmp_path / "a.json"; p.write_text(json.dumps(data)); return str(p)


class TestLoader:
    def test_keeps_only_vehicles_and_maps_by_name(self, coco_file):
        gt = load_coco_vehicles(coco_file)
        assert gt[1].boxes == [(10.0, 10.0, 30.0, 30.0, 2)]
        assert gt[2].boxes == [(30.0, 30.0, 70.0, 70.0, 7)]
        assert len(gt) == 10

    def test_missing_file(self):
        with pytest.raises(FileNotFoundError):
            load_coco_vehicles("nope.json")

    def test_no_vehicle_categories(self, tmp_path):
        p = tmp_path / "b.json"
        p.write_text(json.dumps({"images": [], "annotations": [], "categories": [{"id": 1, "name": "dog"}]}))
        with pytest.raises(ValueError):
            load_coco_vehicles(str(p))


class TestSplits:
    def test_disjoint_complete_reproducible(self):
        ids = list(range(100))
        s1 = make_splits(ids, seed=1); s2 = make_splits(ids, seed=1)
        assert s1 == s2
        allv = s1["train"] + s1["val"] + s1["test"]
        assert sorted(allv) == ids and len(set(allv)) == 100
        assert (len(s1["train"]), len(s1["val"]), len(s1["test"])) == (70, 15, 15)

    def test_bad_ratios(self):
        with pytest.raises(ValueError):
            make_splits([1, 2, 3], ratios=(0.5, 0.5, 0.5))

    def test_save_load_roundtrip(self, tmp_path):
        s = make_splits(list(range(20)))
        p = tmp_path / "s.json"; save_splits(s, str(p), seed=42)
        assert load_splits(str(p)) == s


class TestMetrics:
    GT = {1: [(10, 10, 50, 50, 2)], 2: [(0, 0, 30, 30, 7)]}

    def test_perfect_predictions(self):
        preds = {1: [((10, 10, 50, 50), 2, 0.9)], 2: [((0, 0, 30, 30), 7, 0.8)]}
        r = evaluate_detections(preds, self.GT)
        assert r["mAP"] == pytest.approx(1.0) and r["f1"] == pytest.approx(1.0)

    def test_no_predictions(self):
        r = evaluate_detections({1: [], 2: []}, self.GT)
        assert r["mAP"] == 0.0 and r["recall"] == 0.0

    def test_wrong_class_is_not_a_match(self):
        preds = {1: [((10, 10, 50, 50), 7, 0.9)]}
        r = evaluate_detections(preds, self.GT)
        assert r["recall"] == 0.0 and r["precision"] == 0.0

    def test_duplicate_detection_counts_as_false_positive(self):
        preds = {1: [((10, 10, 50, 50), 2, 0.9), ((10, 10, 50, 50), 2, 0.8)]}
        r = evaluate_detections(preds, {1: self.GT[1]})
        assert r["per_class"][2]["tp"] == 1 and r["per_class"][2]["fp"] == 1

    def test_ap_hand_computed(self):
        # ranked TP, FP, TP with 2 GT -> precision envelope gives AP = 0.5*1 + 0.5*(2/3)
        assert average_precision([1, 0, 1], 2) == pytest.approx(0.5 + 0.5 * 2 / 3)

    def test_ap_no_ground_truth_is_nan(self):
        import math
        assert math.isnan(average_precision([1], 0))
