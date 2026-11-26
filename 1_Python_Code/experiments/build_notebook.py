"""Build and execute RATFA_Experiments.ipynb (all outputs embedded)."""
import os

import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [
    md("# RATFA: Reliability-Aware Temporal Fuzzy Adaptation\n"
       "Implementation, training and testing notebook.\n\n"
       "Pipeline: video frame -> scene-condition estimation (q) -> F-YOLO detection (GPBlock, FFM, FA) -> "
       "detection reliability (r) -> temporal consistency (tau) -> confidence gate gamma = q*r*tau -> "
       "gated fuzzy-membership update -> ByteTrack -> detections with track IDs.\n\n"
       "Full re-run from scratch (about 25 min on a CPU laptop):\n"
       "```\npython experiments/download_datasets.py\npython experiments/train_detector.py --epochs 5\n"
       "python experiments/train_scene_classifier.py\npython experiments/calibrate_conditions.py\n"
       "python experiments/run_experiments.py\npython experiments/make_figures.py\n```"),
    code("import os, sys, json, glob\nos.chdir(os.path.abspath('..')) if os.path.basename(os.getcwd()) == 'experiments' else None\n"
         "sys.path.insert(0, os.getcwd()); os.environ.setdefault('YOLO_CONFIG_DIR', os.path.join(os.getcwd(), '.ultralytics'))\n"
         "import numpy as np, pandas as pd, cv2, matplotlib.pyplot as plt\nfrom IPython.display import Image, display, Markdown\n"
         "pd.set_option('display.precision', 3); print(os.getcwd())"),
    md("## 1. Datasets\nUA-DETRAC (vehicle boxes + track IDs) and BDD100K val (real weather / time-of-day labels)."),
    code("tr = glob.glob('datasets/UA-DETRAC/train/*/*.jpg'); te = glob.glob('datasets/UA-DETRAC/test/*/*.jpg')\n"
         "bdd = json.load(open('datasets/BDD100K/bdd100k_ratfa_subset.json'))\n"
         "print('UA-DETRAC train frames:', len(tr), '| sequences:', sorted(os.listdir('datasets/UA-DETRAC/train')))\n"
         "print('UA-DETRAC test frames :', len(te), '| sequences:', sorted(os.listdir('datasets/UA-DETRAC/test')))\n"
         "print('BDD100K images per condition:', pd.Series([r['condition'] for r in bdd]).value_counts().to_dict())"),
    code("from experiments.corruptions import apply, CONDITIONS\nimg = cv2.imread(sorted(te)[0])\n"
         "fig, ax = plt.subplots(1, 4, figsize=(16, 2.6))\n"
         "for a, c in zip(ax, CONDITIONS):\n    a.imshow(apply(img, c, seed=0)[:, :, ::-1]); a.set_title(c); a.axis('off')\nplt.show()"),
    md("## 2. F-YOLO model (fuzzy modules) and training results"),
    code("from ultralytics import YOLO\nfrom detector.fyolo_model import fuzzy_sets\n"
         "fy = YOLO('checkpoints/fyolo_detrac.pt')\nsets = fuzzy_sets(fy.model)\n"
         "print('Adaptable fuzzy units:', len(sets))\nfor n, s in sets.items():\n"
         "    print(f'{n:22s} centres={np.round(s.centres.detach().numpy(), 3)} sigmas={np.round(s.sigmas.detach().numpy(), 3)}')"),
    code("rows = [json.load(open(f'outputs/training/{k}_val_metrics.json')) for k in ('baseline', 'fyolo')]\n"
         "display(pd.DataFrame(rows)); display(Image('outputs/figures/training_curves.png'))"),
    md("## 3. Scene-condition classifier (q_t)"),
    code("sc = json.load(open('outputs/scene_analysis/scene_classifier_results.json'))\n"
         "for k in ('bdd100k_real', 'ua_detrac_synthetic'):\n    print(k, 'accuracy =', round(sc[k]['accuracy'], 3), 'n =', sc[k]['n_test'])\n"
         "    display(pd.DataFrame(sc[k]['confusion_matrix'], index=CONDITIONS, columns=CONDITIONS))"),
    md("## 4. Live RATFA demo on one test sequence (fog)\nThe controller computes q, r, tau and gamma every frame and updates the fuzzy memberships."),
    code("import experiments.run_experiments as E\nfrom scene_analysis.scene_classifier import SceneClassifier\n"
         "models = E.load_models(); cfg = json.load(open('checkpoints/ratfa_condition_params.json'))\n"
         "clf = SceneClassifier.load('checkpoints/scene_classifier.pkl')\n"
         "frames, gt = E.load_sequence(sorted(glob.glob('datasets/UA-DETRAC/test/*'))[1])\n"
         "det, trk, lat, log = E.run_sequence('F-YOLO + RATFA', models, cfg, clf, frames[:60], gt, 'fog', 1)\n"
         "g = pd.DataFrame(log)[['frame', 'q', 'r', 'tau', 'gamma', 'applied', 'delta_norm']]\n"
         "display(g.head(10)); print('detection:', E.detection_metrics(det)); print('tracking:', E.tracking_metrics([trk]))"),
    code("img = E.apply(cv2.imread(frames[59]), 'fog', seed=1059)\nb = E.predict(models[1], img)\n"
         "for x1, y1, x2, y2 in b.xyxy[b.conf > 0.25].astype(int):\n    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)\n"
         "plt.figure(figsize=(10, 5.6)); plt.imshow(img[:, :, ::-1]); plt.axis('off'); plt.title('F-YOLO + RATFA, fog'); plt.show()"),
    md("## 5. Main results: UA-DETRAC test (detection, tracking, calibration, runtime)"),
    code("d = pd.DataFrame(json.load(open('outputs/results/ua_detrac_results.json')))\n"
         "cols = ['condition', 'method', 'AP50', 'precision', 'recall', 'F1', 'ECE', 'MOTA', 'IDF1', 'ID_switches', 'FPS', 'latency_ms']\n"
         "display(d[cols])"),
    code("display(d.groupby('method')[['AP50', 'F1', 'ECE', 'MOTA', 'IDF1', 'FPS']].mean().loc[list(dict.fromkeys(d.method))])"),
    code("for m in ('AP50', 'MOTA', 'IDF1', 'ECE'):\n    display(Image(f'outputs/figures/detrac_{m}.png'))\ndisplay(Image('outputs/figures/gate_trace.png'))"),
    md("## 6. Cross-domain: BDD100K (real rain / fog / night images)"),
    code("b = pd.DataFrame(json.load(open('outputs/results/bdd100k_results.json')))\n"
         "display(b[['condition', 'method', 'n_images', 'AP50', 'precision', 'recall', 'F1', 'ECE']]); display(Image('outputs/figures/bdd_AP50.png'))"),
    md("## 7. Findings\n"
       "* The F-YOLO re-implementation (GPBlock + FFM + FA) reached val mAP50 0.876 vs 0.865 for the plain YOLOv8n "
       "and was better on BDD100K clear/rain, but slightly worse on the UA-DETRAC test set.\n"
       "* The confidence gate gamma = q*r*tau behaves as designed: it is high on stable frames and drops when "
       "reliability or temporal consistency drop, which reduces or skips the update.\n"
       "* With this small training budget the RATFA adaptation changes the fuzzy parameters but its effect on the "
       "metrics is small (within about +/-0.01 AP50 / MOTA), so no significant improvement is claimed.\n"
       "* Limitations: official F-YOLO code/weights are not public (re-implemented on YOLOv8n); small data subsets; "
       "rain/fog/dark on UA-DETRAC are synthetic corruptions; BDD100K has only 13 foggy val images; CPU-only training (5 epochs)."),
]
nb = nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}})
ExecutePreprocessor(timeout=1200, kernel_name="python3").preprocess(nb, {"metadata": {"path": ROOT}})
nbf.write(nb, os.path.join(ROOT, "RATFA_Experiments.ipynb"))
print("written RATFA_Experiments.ipynb")
