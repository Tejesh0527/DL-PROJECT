# RATFA: Submission Files

| Folder | Requirement | Contents |
|---|---|---|
| `1_Python_Code/` | Python Code (.py) | All source code: `detector/fyolo_model.py` (F-YOLO fuzzy modules GPBlock, FFM, FA), `fuzzy/ratfa_controller.py` (RATFA controller, gate gamma = q*r*tau), `scene_analysis/` (scene condition, q), `detector/vehicle_detection.py` (reliability, r), `tracking/` (temporal consistency, tau), `experiments/` (download, training, testing, figures), `tests/` (168 unit tests), `config/`, `checkpoints/` (trained models), `requirements.txt` |
| `2_Jupyter_Notebook/` | Jupyter Notebook (.ipynb) | `RATFA_Experiments.ipynb` (already executed, all outputs saved) and `outputs/` (result tables, figures, tracking videos, training curves, logs) |
| `3_Datasets/` | Datasets | UA-DETRAC and BDD100K subsets used, plus `README_DATASETS.md` (sources, licences, splits, seeds, reproduction steps) |

To re-run the notebook: copy `RATFA_Experiments.ipynb` and `outputs/` into `1_Python_Code/`, copy the
datasets into `1_Python_Code/datasets/`, install `requirements.txt`, then open the notebook from `1_Python_Code/`.
