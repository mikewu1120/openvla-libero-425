# Language Robustness in OpenVLA (LIBERO-Spatial · 12 GB RTX 4080 Laptop)

Local reproduction, QLoRA fine-tuning, and language-paraphrase robustness evaluation of **OpenVLA-7B** on **LIBERO-Spatial** using a single 12 GB consumer GPU (RTX 4080 Laptop GPU).

> **Note:** Detailed Chinese setup and training logs are also available in [README_复现.md](README_复现.md) and [README_训练.md](README_训练.md).

---

## Key Results Summary

Across **270 simulation rollouts** (3 LIBERO-Spatial tasks × 10 paired initial states × 3 instruction phrasings × 3 models):

| Model | Original | Paraphrase 1 (P1) | Paraphrase 2 (P2) | Gap (Original − Mean P1/P2) |
|---|---:|---:|---:|---:|
| Base (`openvla-7b-finetuned-libero-spatial`) | 73.3% | 50.0% | 60.0% | 18.3 pp |
| Original-Only QLoRA (`qlora_language_3tasks`) | 80.0% | 63.3% | 60.0% | 18.3 pp |
| Paraphrase-Augmented QLoRA (`qlora_language_augmented`) | 86.7% | 63.3% | 80.0% | 15.0 pp |

- Compared with original-only QLoRA, the exploratory augmented QLoRA run shifts the original-vs-paraphrase gap by **−3.3 pp** (95% paired CI: **[−13.3, +6.7] pp**, which includes zero).
- Open [final_project/index.html](final_project/index.html) for the interactive results & rollout video viewer, [final_project/OpenVLA_Report.pdf](final_project/OpenVLA_Report.pdf) for the written report, and [final_project/presentation.html](final_project/presentation.html) / [final_project/OpenVLA_Final.pptx](final_project/OpenVLA_Final.pptx) for the slides.

---

## Repository Contents

- `run_libero_4080.py` — 4-bit NF4 OpenVLA inference & LIBERO-Spatial simulation runner (`--sim-only`, `--adapter`, `--task-id`, etc.)
- `train_qlora_4080.py` — Reproducible QLoRA training script (`r=8`, `alpha=16`, `q_proj`/`v_proj`, batch=1, grad accum=4, NF4 + SDPA + gradient checkpointing)
- `download_model.py` — Resumable, SHA256-verified downloader for `openvla/openvla-7b-finetuned-libero-spatial`
- `prepare_training_data.py` — Extracts single-task `.npz` demonstrations from verified RLDS shards
- `apply_windows_fixes.py` — Applies `robosuite==1.4.1` DLL/GLFW fixes and LIBERO `env_wrapper.py` reset exception fix on Windows
- `requirements-windows.lock.txt` — Pinned Python 3.10 dependency lockfile
- `datasets/` — Pre-extracted `.npz` training & validation episodes ready for QLoRA training:
  - `datasets/training_subset/` — Task 0 subset (5 train / 2 val trajectories; 738 frames)
  - `datasets/language_experiment/` — Balanced 3-task subset (10 train / 2 val trajectories per task; 3,467 train / 803 val frames)
- `training_runs/` — Saved LoRA adapters (`adapter_model.safetensors`), `metrics.jsonl`, `summary.json`, and training logs/plots
- `results/` — Smoke-test simulation rollout videos, action traces, and summaries
- `final_project/` — Complete presentation & report package (`index.html`, `presentation.html`, `OpenVLA_Final.pptx`, `OpenVLA_Slides.pdf`, `OpenVLA_Report.pdf`, `demos/`, `videos/`, `data/`, and `source.zip` containing `language_experiment/` and `improvement/` evaluation/augmentation scripts)
- `openvla/` & `LIBERO/` — Git submodules pinned to upstream commits `c8f03f48af692657d3060c19588038c7220e9af9` and `8f1084e3132a39270c3a13ebe37270a43ece2a01`

### Excluded Large Files (Not Stored in Git)
The following files are excluded via `.gitignore` due to GitHub file/repo size limits and can be restored using the scripts below:
- `.venv/` (~8.8 GB Python environment)
- `models/openvla-7b-finetuned-libero-spatial/` (~14 GB base weights)
- `datasets/training_subset/raw/*.tfrecord-*` (>100 MB raw RLDS shards; checksums preserved in `datasets/training_subset/raw/manifest.json`)

---

## How to Reproduce & Train on These Datasets

### 1. Clone Repository & Submodules
```bash
git clone --recurse-submodules https://github.com/mikewu1120/openvla-libero-425.git
cd openvla-libero-425
```
If you already cloned without `--recurse-submodules`:
```bash
git submodule update --init --recursive
```

### 2. Set Up Python 3.10 Environment
On Windows (PowerShell):
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.lock.txt --extra-index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe apply_windows_fixes.py
```
> On Linux, `apply_windows_fixes.py` is a no-op for `robosuite`, and you should apply the 2-line fix in `LIBERO/libero/libero/envs/env_wrapper.py` (removing `finally: continue` under `except RandomizationError`) if needed.

### 3. Download the Base 7B Checkpoint
```powershell
.\.venv\Scripts\python.exe download_model.py
```
This downloads `openvla/openvla-7b-finetuned-libero-spatial` (revision `962318cec55ac10993ff0f5f43eda9a270b4c873`) into `models/openvla-7b-finetuned-libero-spatial/` with byte-range resume and SHA256 verification.

### 4. Prepare or Use the Training Datasets
The preprocessed `.npz` episodes are **already included** in `datasets/training_subset/` and `datasets/language_experiment/`, so you can skip straight to Step 5 unless you want to rebuild them from scratch.

To rebuild from raw `openvla/modified_libero_rlds` shards (revision `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551`):
- **Single-task subset** (requires downloading shards `00000` and `00001` listed in `datasets/training_subset/raw/manifest.json` into `datasets/training_subset/raw/`):
  ```powershell
  .\.venv\Scripts\python.exe prepare_training_data.py
  ```
- **3-task language experiment subset** (automatically downloads & verifies required shards):
  Unzip `final_project/source.zip` into the project root to restore `language_experiment/` and `improvement/`, then run:
  ```powershell
  .\.venv\Scripts\python.exe language_experiment\prepare_data.py
  ```

### 5. Run QLoRA Training
> `train_qlora_4080.py` requires CUDA with at least **8 GiB free VRAM** and refuses to overwrite an existing `--output` directory. Peak PyTorch allocation during training is ~5.51 GiB.

1. **Memory & backward-pass sanity probe (1 step):**
   ```powershell
   .\.venv\Scripts\python.exe -u train_qlora_4080.py --steps 1 --accumulation 1 --eval-frames 4 --output training_runs/memory_probe_new
   ```
2. **Single-task QLoRA (50 updates on `datasets/training_subset`):**
   ```powershell
   .\.venv\Scripts\python.exe -u train_qlora_4080.py --steps 50 --data datasets/training_subset --output training_runs/qlora_task0_new
   ```
3. **3-task QLoRA with original instructions (300 updates on `datasets/language_experiment`):**
   ```powershell
   .\.venv\Scripts\python.exe -u train_qlora_4080.py --steps 300 --data datasets/language_experiment --output training_runs/qlora_language_3tasks_new
   ```
4. **3-task QLoRA with paraphrase augmentation (300 updates):**
   First extract `final_project/source.zip` into the project root (restoring `language_experiment/` and `improvement/`). Because `improvement/train_augmented.py` writes to `training_runs/qlora_language_augmented`, move or rename the existing folder if you want to re-run it:
   ```powershell
   .\.venv\Scripts\python.exe -u improvement\train_augmented.py
   ```

### 6. Evaluate Trained Adapters in Simulation
- **Simulation rendering check (no model weights needed):**
  ```powershell
  .\.venv\Scripts\python.exe -u run_libero_4080.py --sim-only --steps 10 --output results/simulation_check
  ```
- **Run base model on Task 0:**
  ```powershell
  .\.venv\Scripts\python.exe -u run_libero_4080.py --task-id 0 --episodes 1 --output results/base_task0
  ```
- **Run a trained LoRA adapter on Task 0:**
  ```powershell
  .\.venv\Scripts\python.exe -u run_libero_4080.py --adapter training_runs/qlora_task0/adapter --output results/qlora_task0_eval
  ```
- **Full 3-model × 3-task × 3-instruction evaluation & analysis** (after extracting `final_project/source.zip`):
  ```powershell
  .\.venv\Scripts\python.exe language_experiment\finish_run.py
  .\.venv\Scripts\python.exe improvement\run_extension.py
  .\.venv\Scripts\python.exe language_experiment\summarize.py
  .\.venv\Scripts\python.exe improvement\analyze_extension.py
  ```

---

## References & Upstream Sources
- [OpenVLA Paper (arXiv:2406.09246)](2406.09246v3.pdf)
- [OpenVLA Official Repository](https://github.com/openvla/openvla)
- [OpenVLA LIBERO-Spatial Checkpoint](https://huggingface.co/openvla/openvla-7b-finetuned-libero-spatial)
- [Modified LIBERO RLDS Dataset](https://huggingface.co/datasets/openvla/modified_libero_rlds)
- [LIBERO Official Repository](https://github.com/Lifelong-Robot-Learning/LIBERO)
