"""Train / test the scene-condition classifier (clear, rain, fog, dark).

Training data:
  * BDD100K real images (70 % stratified split) with weather / time-of-day labels
  * extra synthetic-fog images (BDD100K val has only 13 foggy images)
  * UA-DETRAC training frames: clean + synthetic rain / fog / low light (severity 0.8)
Test data (never seen in training):
  * BDD100K real held-out 30 %
  * UA-DETRAC test frames with the test-severity corruptions (severity 1.0)
"""
import glob
import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.corruptions import apply  # noqa: E402
from scene_analysis.scene_classifier import CONDITIONS, SceneClassifier, scene_feature_vector  # noqa: E402

BDD = os.path.join(ROOT, "datasets", "BDD100K")
OUT = os.path.join(ROOT, "outputs", "scene_analysis")


def main():
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    os.makedirs(OUT, exist_ok=True)
    recs = json.load(open(os.path.join(BDD, "bdd100k_ratfa_subset.json")))
    labels = [r["condition"] for r in recs]
    tr, te = train_test_split(recs, test_size=0.3, random_state=42, stratify=labels)
    X, y = [], []
    for r in tr:
        img = cv2.imread(os.path.join(BDD, "images", r["file"]))
        X.append(scene_feature_vector(img)); y.append(r["condition"])
    clear_tr = [r for r in tr if r["condition"] == "clear"]
    for i, r in enumerate(clear_tr):                       # synthetic fog augmentation
        img = cv2.imread(os.path.join(BDD, "images", r["file"]))
        X.append(scene_feature_vector(apply(img, "fog", seed=1000 + i, severity=0.8))); y.append("fog")
    det_frames = sorted(glob.glob(os.path.join(ROOT, "datasets", "UA-DETRAC", "train", "*", "*.jpg")))
    for i, p in enumerate(det_frames[::3]):
        img = cv2.imread(p)
        for c in CONDITIONS:
            X.append(scene_feature_vector(apply(img, c, seed=2000 + i, severity=0.8))); y.append(c)
    clf = SceneClassifier().fit(np.stack(X), np.array(y))
    clf.save(os.path.join(ROOT, "checkpoints", "scene_classifier.pkl"))

    results = {"n_train": len(y)}
    # Test 1: BDD100K real images
    yt, yp = [], []
    for r in te:
        e, _ = clf.predict_proba_img(cv2.imread(os.path.join(BDD, "images", r["file"])))
        yt.append(r["condition"]); yp.append(CONDITIONS[int(e.argmax())])
    # Test 2: UA-DETRAC test frames with test-severity corruptions
    yt2, yp2 = [], []
    test_frames = sorted(glob.glob(os.path.join(ROOT, "datasets", "UA-DETRAC", "test", "*", "*.jpg")))
    for i, p in enumerate(test_frames[::10]):
        img = cv2.imread(p)
        for c in CONDITIONS:
            e, _ = clf.predict_proba_img(apply(img, c, seed=i, severity=1.0))
            yt2.append(c); yp2.append(CONDITIONS[int(e.argmax())])
    for name, a, b in (("bdd100k_real", yt, yp), ("ua_detrac_synthetic", yt2, yp2)):
        results[name] = {
            "n_test": len(a), "accuracy": accuracy_score(a, b),
            "confusion_matrix": confusion_matrix(a, b, labels=CONDITIONS).tolist(),
            "labels": CONDITIONS,
            "report": classification_report(a, b, labels=CONDITIONS, output_dict=True, zero_division=0)}
        print(name, "accuracy = %.3f" % results[name]["accuracy"])
        print(classification_report(a, b, labels=CONDITIONS, zero_division=0))
    json.dump(results, open(os.path.join(OUT, "scene_classifier_results.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
