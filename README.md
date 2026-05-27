# EigeNet

Code for "EigeNet: Geometry-Informed Multi-Modal Learning for Few-shot Novel View RIR Prediction"

## Requirements

- Linux (recommended)
- CUDA 12.1 (compatible with PyTorch 2.4.0)
- [Conda](https://docs.conda.io/) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html)

## Environment Setup

```bash
cd /path/to/EigeNet
conda env create -f environment.yml
conda activate eigenet
```

If PyTorch does not detect your GPU, install the CUDA 12.1 build manually:

```bash
pip install torch==2.4.0 torchaudio==2.4.0 torchvision==0.19.0 \
  --index-url https://download.pytorch.org/whl/cu121
```

### Environment variables

Set these before training or inference:

```bash
export WORK_DIR=/path/to/EigeNet          # project root
export PYTHONPATH=$WORK_DIR
export PYTHONIOENCODING=UTF-8
export HF_ENDPOINT=https://hf-mirror.com  # optional, Hugging Face mirror
export SWANLAB_API_KEY=your_key           # optional, for SwanLab experiment tracking
```

## Data Preparation

### Data Preprocessing

Use the scripts under `_preprocess/` to generate jsonl files from [AcousticRooms](https://github.com/facebookresearch/AcousticRooms) and [Hear-Anything-Anywhere (DIFFRIR)](https://zenodo.org/records/11195833) datasets. Before running, edit the path variables at the top of each script (`dataset_root`, `output_jsonl`, etc.) to match your local setup.

#### AcousticRooms

Download and unzip `single_channel_ir.zip`, `metadata.zip`, and `depth_map.zip` from the [AcousticRooms repository](https://github.com/facebookresearch/AcousticRooms). The expected layout is:

```
AcousticRooms/
├── depth_map/
│   └── {scene_name}/{scene_id}/{rec_idx}.npy
├── metadata/
│   └── {scene_name}/{scene_id}/S{src_idx:03d}_R{rec_idx:03d}.json
└── single_channel_ir/
    └── {scene_name}/{scene_id}/S{src_idx:03d}_R{rec_idx:03d}_hybrid_IR.wav
```

**Step 1 — Generate training jsonl**

```bash
conda activate eigenet
cd /path/to/EigeNet

# Edit dataset_root and output_jsonl in the script first, then run:
python _preprocess/preprocess_train.py
```

**Step 2 — Generate test jsonl**

```bash
# Edit dataset_root and output_jsonl in the script first, then run:
python _preprocess/preprocess_test.py
```

#### Hear-Anything-Anywhere (HAA)

Download the four base-configuration zip files from [Zenodo](https://zenodo.org/records/11195833): `classroomBase.zip`, `complexBase.zip`, `dampenedBase.zip`, and `hallwayBase.zip`. Each zip contains `RIRs.npy`, `xyzs.npy`, and other raw recordings at **48 kHz**.

The original HAA release does **not** include `single_channel_ir/` or `depth.npy`. Prepare them as follows.

**Step 1 — Prepare depth maps**

HAA does not ship panoramic depth maps. Download them from the [xRIR depth_map folder](https://github.com/DragonLiu1995/xRIR_code/tree/main/sim_to_real/depth_map) and rename/copy each file into the corresponding scene directory as `depth.npy`:

| Scene folder | xRIR source file | Target path |
|--------------|------------------|-------------|
| `classroomBase` | `class_room.npy` | `classroomBase/depth.npy` |
| `complexBase` | `complex_room.npy` | `complexBase/depth.npy` |
| `dampenedBase` | `dampened_room.npy` | `dampenedBase/depth.npy` |
| `hallwayBase` | `hallway.npy` | `hallwayBase/depth.npy` |

Example:

```bash
# after downloading the four .npy files from xRIR
cp class_room.npy   /path/to/haa/classroomBase/depth.npy
cp complex_room.npy /path/to/haa/complexBase/depth.npy
cp dampened_room.npy /path/to/haa/dampenedBase/depth.npy
cp hallway.npy      /path/to/haa/hallwayBase/depth.npy
```

**Step 2 — Export single-channel RIR wav files**

The raw HAA dataset stores monaural RIRs in `RIRs.npy` with shape `(N, T)` at 48 kHz. Use `export_single_channel_ir_for_all_scenes()` (called automatically when `EXPORT_RIR = True`) to write each row to `single_channel_ir/{scene_name}_{src_idx}.wav`:

```bash
conda activate eigenet
cd /path/to/EigeNet

# Edit DATASET_ROOT and output_jsonl in the script first, then run:
python _preprocess/preprocess_haa.py
```

After preprocessing, each scene should look like:

```
haa/
├── classroomBase/
│   ├── RIRs.npy                  # from Zenodo
│   ├── xyzs.npy                  # from Zenodo (source microphone locations)
│   ├── depth.npy                 # from xRIR_code (renamed)
│   └── single_channel_ir/        # exported by preprocess_haa.py
│       ├── classroomBase_0.wav
│       ├── classroomBase_1.wav
│       └── ...
├── complexBase/
├── dampenedBase/
└── hallwayBase/
```

**Step 2 — Split into train / test jsonl**

```bash
python _preprocess/split_haa.py
```

This script splits `haa.jsonl` into train and test sets with an 80/20 ratio **per scene**, adds a `split` field to each entry, and converts absolute file paths to paths relative to `HAA_ROOT`.

After preprocessing, update `HAA_ROOT` in `models/dataset/haa_dataset.py` and `haa_dataset_test.py` to match your data location.

---

### 1. AcousticRooms

Update paths in `egs/rir/EigeNet/EigeNet.json`:

| Field | Description |
|-------|-------------|
| `dataset.rir_list` | Training jsonl file list |
| `dataset.test_rir_list` | Test jsonl file list |
| `dataset.depth_map_root` | Root directory for depth maps |
| `dataset.rir_root` | Root directory for RIR audio |
| `dataset.metadata_root` | Root directory for scene metadata |

Generate jsonl index files with `_preprocess/preprocess_train.py` and `_preprocess/preprocess_test.py`

### 2. Fine-tuning data (HAA)

Update `egs/rir/EigeNet/EigeNet_finetune_haa.json`:

| Field | Description |
|-------|-------------|
| `dataset.haa_train_path` | HAA training jsonl |
| `dataset.haa_test_path` | HAA test jsonl |

Generate jsonl index files with `_preprocess/preprocess_haa.py` and `_preprocess/split_haa.py` (see [Data Preprocessing](#data-preprocessing) above). Expected outputs:

- `data/haa_train.jsonl`
- `data/haa_test.jsonl`

Set `HAA_ROOT` in `models/dataset/haa_dataset.py` and `haa_dataset_test.py` to your local HAA data root.

### 3. Pre-trained DAC

Both config files require [DAC weights](https://github.com/descriptinc/descript-audio-codec). Set `model.DAC.path` to your local checkpoint:

```json
"DAC": {
  "path": "/path/to/dac/weights_16k.pth"
}
```

## Pre-trained Checkpoints

We provide our trained EigeNet checkpoints on Google Drive:

**Download:** [Google Drive link](https://drive.google.com/drive/folders/1D4M3-1wEJJBC4WEjNhNjThfDzgN0lg3j?usp=sharing)

Download `EigeNet_ckpt.tar.gz` and extract it under the project root:

```bash
cd /path/to/EigeNet
mkdir -p ckpts
tar -xzf EigeNet_ckpt.tar.gz -C ckpts/
```

## Training (AcousticRooms)

Script: `egs/rir/EigeNet/EigeNet.sh`

```bash
conda activate eigenet
cd /path/to/EigeNet

# Run directly (adjust GPU count for your cluster)
bash egs/rir/EigeNet/EigeNet.sh
```

Checkpoints are saved to `ckpts/EigeNet/{exp_name}/checkpoint/` by default (controlled by `log_dir` in the config).

## Fine-tuning (HAA)

Script: `egs/rir/EigeNet/EigeNet_finetune_haa.sh`

```bash
conda activate eigenet
cd /path/to/EigeNet
bash egs/rir/EigeNet/EigeNet_finetune_haa.sh
```

Before fine-tuning, verify:

1. HAA data paths in `EigeNet_finetune_haa.json` are correct
2. `--checkpoint_path` in EigeNet_finetune_haa.sh points to a checkpoint from AcousticRooms training

## Inference

Inference scripts live in `models/EigeNet/src/` and load the model and DAC codec via `InferencePipeline`. Edit `cfg`, `ckpt`, and related parameters in the script's `__main__` block before running.

### AcousticRooms test set

```bash
conda activate eigenet
cd /path/to/EigeNet

python models/EigeNet/src/infer_ar.py
```

### HAA test set

```bash
conda activate eigenet
cd /path/to/EigeNet

python models/EigeNet/src/infer_haa.py
```

