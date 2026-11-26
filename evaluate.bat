@echo off
REM RATFA Evaluation Script
echo ============================================
echo RATFA Evaluation
echo ============================================

REM Activate environment
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"
call venv\Scripts\activate.bat

echo.
echo Checking for evaluation setup...

REM Check if splits exist
if not exist "outputs\datasets\splits.json" (
    echo Creating train/val/test splits...
    python -c "
from data_tools.coco_dataset import make_splits, save_splits
import json

# Load annotations
annotations = 'data\sample_dataset\instances_sample.json'
with open(annotations) as f:
    data = json.load(f)

gt = {}
for img in data['images']:
    gt[img['id']] = {'file_name': img['file_name'], 'boxes': []}

for ann in data['annotations']:
    if img_id := ann['image_id'] in gt:
        gt[img_id]['boxes'].append(ann['bbox'])

# Create 70/15/15 splits
ids = list(gt.keys())
splits = make_splits(ids, seed=42)
save_splits(splits, 'outputs\datasets\splits.json', seed=42)

print('Splits created:', {k: len(v) for k, v in splits.items()})
"
)

REM Run evaluation
echo.
echo Running evaluation on test split...
python evaluate.py --config config\config.yaml ^
    --annotations data\sample_dataset\instances_sample.json ^
    --images-dir data\sample_dataset\images ^
    --split-file outputs\datasets\splits.json ^
    --split test ^
    --output outputs\evaluation

echo.
echo Evaluation Complete.
echo Results saved to outputs\evaluation\

pause