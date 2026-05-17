# HW2: Deep Learning and Spatial Intelligence

This repository contains the code, experimental report, logs, and visualizations for three tasks in HW2:

1. Flower classification with fine-tuned CNNs on the 102 Category Flower Dataset
2. Road vehicle detection, multi-object tracking, and line-crossing counting with YOLOv8 + ByteTrack
3. Semantic segmentation with a hand-written U-Net and custom Dice Loss on the Stanford Background Dataset

The repository is organized as follows:

```text
hw2/
|-- data/
|   |-- 102flowers/
|   |-- trafic_data/
|   `-- iccv09Data/
|-- figures/
|-- task1/
|-- task2/
|-- task3/
|-- report.md
`-- README.md
```

## 1. Environment Setup

The experiments were developed and tested with the following environment:

- Python 3.10
- PyTorch 2.7.1+cu128
- torchvision
- ultralytics 8.4.41
- timm 1.0.26
- swanlab 0.7.16
- numpy
- scipy
- Pillow
- opencv-python
- pyyaml

### 1.1 Recommended conda environment

```bash
conda create -n spatial-ai-hw2 python=3.10 -y
conda activate spatial-ai-hw2
```

### 1.2 Install PyTorch

If you are using CUDA 12.8, install PyTorch with:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

If your CUDA version is different, please install a matching PyTorch build from the official PyTorch website.

### 1.3 Install remaining dependencies

```bash
pip install -r requirements.txt
```

For ByteTrack, `ultralytics` may automatically install `lap` on the first run. You can also install it manually in advance:

```bash
pip install lap
```

### 1.4 Verify the environment

```bash
python -c "import torch, torchvision, ultralytics, timm, swanlab, cv2, scipy; print('Environment OK')"
```

## 2. Important Path Configuration

The repository has been updated to use repository-relative paths by default.

If you keep the default directory structure after cloning, the training and testing scripts should work without manually editing hard-coded absolute paths.

Expected root structure:

```text
hw2/
|-- data/
|-- task1/
|-- task2/
|-- task3/
`-- README.md
```

You only need to modify paths manually in the following situations:

- your datasets are not stored under the default `data/` directory
- you want to save outputs to a custom location
- you want to use a different test video or model checkpoint path

Recommended approach:

1. Clone the repository.
2. Place the datasets under the root `data/` directory.
3. Run the scripts from their corresponding task directories.

If your data is stored elsewhere, edit the corresponding path variables in the task scripts.

## 3. Dataset Preparation

Place all datasets under the root `data/` directory.

Expected structure:

```text
data/
|-- 102flowers/
|   |-- jpg/
|   |-- imagelabels.mat
|   `-- setid.mat
|-- trafic_data/
|   |-- train/
|   |   |-- images/
|   |   `-- labels/
|   `-- valid/
|       |-- images/
|       `-- labels/
`-- iccv09Data/
    |-- images/
    `-- labels/
```

### 3.1 Task 1 dataset

Task 1 expects the Oxford 102 Flowers dataset in:

```text
data/102flowers/
```

Required files:

- `jpg/`
- `imagelabels.mat`
- `setid.mat`

### 3.2 Task 2 dataset

Task 2 expects the Road Vehicle Images dataset in:

```text
data/trafic_data/
```

Required structure:

- `train/images`
- `train/labels`
- `valid/images`
- `valid/labels`

The label format must be YOLO detection format.

### 3.3 Task 3 dataset

Task 3 expects the Stanford Background Dataset in:

```text
data/iccv09Data/
```

Required structure:

- `images/`
- `labels/`

The code reads `.regions.txt` files from the `labels/` directory.

## 4. Task 1: Flower Classification

Task 1 code is under `task1/`.

### 4.1 Training all preset experiments

The default `task1/train.py` script runs several experiments sequentially:

- ResNet-18, pretrained, `lr=5e-4`
- ResNet-18, pretrained, `lr=5e-3`
- ResNet-18, pretrained, `epochs=50`
- ResNet-18, pretrained, `batch_size=64`
- ResNet-34, pretrained

Run from the `task1/` directory:

```bash
python train.py
```

This is equivalent to:

```bash
python train.py --mode preset
```

### 4.2 Running a single custom experiment

If you want to run one specific experiment instead of all preset experiments, use CLI arguments.

The following commands should also be executed inside the `task1/` directory.

Example: ResNet-18 pretrained baseline

```bash
python train.py --mode single --model-name resnet18 --pretrained --lr 1e-3 --fc-lr 1e-2 --epochs 30 --batch-size 32 --experiment-name resnet18_baseline
```

Example: ResNet-18 + SE

```bash
python train.py --mode single --model-name resnet18_se --pretrained --lr 1e-3 --fc-lr 1e-2 --epochs 30 --batch-size 32 --experiment-name resnet18_se_custom
```

Example: training from scratch

```bash
python train.py --mode single --model-name resnet18 --no-pretrained --lr 1e-3 --epochs 30 --batch-size 32 --experiment-name resnet18_scratch
```

Useful Task 1 CLI arguments:

- `--data-dir`
- `--mode {preset,single}`
- `--model-name`
- `--pretrained` / `--no-pretrained`
- `--batch-size`
- `--lr`
- `--fc-lr`
- `--epochs`
- `--optimizer`
- `--scheduler`
- `--weight-decay`
- `--project`
- `--experiment-name`
- `--save-dir`

Supported model names in `task1/models.py`:

- `resnet18`
- `resnet34`
- `resnet18_se`
- `resnet18_cbam`
- `vit_tiny`
- `swin_tiny`

### 4.3 Evaluation behavior

Task 1 does not provide a separate standalone test script. Evaluation is performed automatically at the end of training:

1. The best checkpoint is selected by validation accuracy.
2. That checkpoint is reloaded.
3. The model is evaluated on the official test split.

### 4.4 Output files

Task 1 saves outputs to:

- `task1/checkpoints/`
- `task1/checkpoints/results/`
- `task1/swanlog/`

Artifacts include:

- model checkpoints (`*_best.pth`)
- experiment summaries (`*.json`)
- SwanLab logs for loss/accuracy curves

### 4.5 View Task 1 logs

```bash
swanlab watch task1/swanlog
```

## 5. Task 2: Detection, Tracking, and Counting

Task 2 code is under `task2/`.

### 5.1 Generate or verify the dataset YAML

If needed, regenerate `vehicle.yaml`:

```bash
python create_yaml.py
```

This script writes the dataset config used by YOLOv8.

### 5.2 Train the YOLOv8 model

Run from the `task2/` directory:

```bash
python train_yolo.py
```

This script:

1. Loads `yolov8n.pt`
2. Trains on `vehicle.yaml`
3. Validates the model after training
4. Prints `mAP50` and `mAP50-95`

### 5.3 Detection outputs

Training results are saved to:

```text
task2/runs/yolov8n_vehicle/
```

Important files include:

- `weights/best.pt`
- `weights/last.pt`
- `results.csv`
- `results.png`
- `confusion_matrix.png`
- `BoxPR_curve.png`

### 5.4 Prepare a test video

For tracking and line counting, place a test video in `task2/`.

The script will prioritize these filenames:

- `test_video.mp4`
- `clip_0.mp4`
- `traffic.mp4`
- `sample_5s.mp4`

If none of these exist, the script will search other video files under `task2/` and `data/trafic_data/`.

### 5.5 Run tracking and line-crossing counting

Run from the `task2/` directory:

```bash
python track_and_count.py
```

This script will:

1. Load the trained detector from `runs/yolov8n_vehicle/weights/best.pt`
2. Run YOLOv8 tracking with ByteTrack
3. Save a tracking video with bounding boxes, class labels, and tracking IDs
4. Automatically select a dense 4-frame window for occlusion analysis
5. Save annotated occlusion frames
6. Automatically choose a counting line
7. Save a line-crossing counting video
8. Save a summary JSON file

### 5.6 Task 2 outputs

Outputs are saved to:

```text
task2/output/
```

Important files:

- `tracked_video.mp4`
- `counting_video.mp4`
- `summary.json`
- `occlusion_frames/frame_*.jpg`

### 5.7 Optional video preprocessing

If you want to trim or resize a raw video before testing, you can use `ffmpeg`. Example:

```bash
ffmpeg -y -ss 0 -t 8 -i input.mp4 -vf "scale=1280:-2" -c:v libx264 -preset veryfast -crf 23 -an task2/test_video.mp4
```

## 6. Task 3: U-Net Segmentation and Dice Loss

Task 3 code is under `task3/`.

### 6.1 Train all preset experiments

The default `task3/train.py` runs three loss settings sequentially:

- Cross-Entropy
- Dice Loss
- Combined Loss (`CE + Dice`)

Run from the `task3/` directory:

```bash
python train.py
```

This is equivalent to:

```bash
python train.py --mode preset
```

### 6.2 Run a single custom experiment

Use CLI arguments for a single custom configuration.

The following commands should also be executed inside the `task3/` directory.

Example: Cross-Entropy only

```bash
python train.py --mode single --loss-type ce --batch-size 4 --lr 1e-3 --epochs 20 --img-size 320 --experiment-name unet_ce_custom
```

Example: Combined loss

```bash
python train.py --mode single --loss-type combined --batch-size 4 --lr 1e-3 --epochs 20 --img-size 320 --experiment-name unet_combined_custom
```

Supported loss types:

- `ce`
- `dice`
- `combined`

Useful Task 3 CLI arguments:

- `--data-dir`
- `--mode {preset,single}`
- `--loss-type`
- `--batch-size`
- `--lr`
- `--epochs`
- `--img-size`
- `--project`
- `--experiment-name`
- `--save-dir`

### 6.3 Evaluation behavior

Task 3 evaluates on the validation set during training and keeps the best checkpoint according to validation `mIoU`.

### 6.4 Output files

Task 3 saves outputs to:

- `task3/checkpoints/`
- `task3/checkpoints/results/`
- `task3/swanlog/`

Artifacts include:

- best checkpoints
- experiment summaries
- SwanLab logs for training/validation curves

### 6.5 View Task 3 logs

```bash
swanlab watch task3/swanlog
```

## 7. Report and Visualizations

Main report file:

```text
report.md
```

Pre-generated figures:

- `figures/task1_summary.png`
- `figures/task2_yolo_curves.png`
- `figures/task2_occlusion_grid.png`
- `figures/task3_miou_comparison.png`

Task 2 also includes video outputs that can be embedded in a submission package or linked externally.

## 8. Quick Start Summary

If the datasets are already placed correctly and the paths have been updated, the fastest way to reproduce the full pipeline is:

### Task 1

```bash
python task1/train.py
```

### Task 2

```bash
python task2/train_yolo.py
python task2/track_and_count.py
```

### Task 3

```bash
python task3/train.py
```

## 9. Notes

1. A `requirements.txt` file is included, but PyTorch should still be installed separately if you need a specific CUDA build.
2. The main scripts now use repository-relative paths by default.
3. Task 1 and Task 3 use SwanLab in local mode. Logs are stored in the repository rather than uploaded to a remote server.
4. Task 2 tracking results depend heavily on the quality and viewpoint of the test video.

## 10. Contact / Submission Checklist

Before submission, make sure the following are complete:

- datasets are placed correctly
- custom paths are updated if you are not using the default `data/` layout
- training commands run successfully
- tracking test video is available
- `report.md` has filled-in cover information
- Github repository link is added to the report
- model weight download link is added to the report
