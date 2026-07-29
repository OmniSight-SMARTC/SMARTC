# SMARTC: Semantic Memory-Based Anomaly Recognition Text-Guided Classifier

Anonymous code repository for double-blind review. This repository contains the six Kaggle notebooks (three per dataset) used to produce every table, figure, and reported number in the paper.

> **Anonymity notice:** This repository has been sanitized for double-blind review. Commit history, usernames, and any identifying paths have been removed. After acceptance, this anonymous link will be replaced with the authors' official repository in the camera-ready version.

---

## 1. Overview

SMARTC is a three-stage pipeline for industrial anomaly detection and open-vocabulary semantic classification:

| Stage | Name | Role | Notebook prefix |
|---|---|---|---|
| 1 | Memory-Guided Spatial Localization | PatchCore-based defect localization and coreset memory bank construction | `NB0` |
| 2 | Structurally-Preserved Semantic ROI Extraction | Tight/context crop extraction around localized defects | `NB1` |
| 3 | Entropy-Normalized Semantic Arbitration | Dual-stream CLIP / AnomalyCLIP classification with entropy-gated routing | `NB2` |

Each stage is run once per dataset (VisA, DAGM), giving six notebooks total:

```
notebooks/
├── VisA_NB0_Research.ipynb      # VisA — Stage 1
├── VisA_NB1_Research.ipynb      # VisA — Stage 2
├── VisA_NB2_Research.ipynb      # VisA — Stage 3
├── DAGM_NB0_Research.ipynb      # DAGM — Stage 1
├── DAGM_NB1_Research.ipynb      # DAGM — Stage 2
└── DAGM_NB2_Research.ipynb      # DAGM — Stage 3
```

Notebooks must be run in order (`NB0 → NB1 → NB2`) per dataset, since each stage consumes the artifacts produced by the previous one.

---

## 2. Environment Setup

**Framework:** PyTorch 2.10.0+cu128, Torchvision 0.25.0+cu128 (confirmed via the environment-check cell run at the top of all six notebooks)

**Hardware used for reported results:** Single NVIDIA Tesla T4 GPU (~15.64 GB VRAM, decimal-GB basis; the same card reports ~14.56 GiB when measured via `total_memory / 1024**3` in a different notebook — same physical hardware, two unit conventions, not a discrepancy), Kaggle cloud environment, 4 CPU cores.

**Key dependencies (as actually installed/imported across the six notebooks):**
```
torch==2.10.0+cu128
torchvision==0.25.0+cu128
pytorch-lightning==2.0.0
anomalib==1.0.1
timm
scikit-learn
opencv-python
faiss-cpu
open_clip_torch                          # Standard CLIP (ViT-B/32)
git+https://github.com/openai/CLIP.git   # OpenAI CLIP (NB2, Cell 0)
AnomalyCLIP                               # cloned from github.com/zqhang/AnomalyCLIP (ViT-L/14)
rembg[gpu]                                 # background removal — VisA preprocessing only (NB0, Cell 8)
ftfy
regex
tqdm
Pillow
pandas
matplotlib
seaborn
psutil                                     # memory-feasibility check (VisA NB0, Cell 5)
scipy                                       # Mahalanobis distance, Spearman correlation
pyarrow                                     # Parquet serialization (NB2, Cell 13)
thop
```

**Determinism / seeds:**
- All experiments use a single fixed seed: `seed = 42`
- Seed is frozen across **PyTorch, NumPy, CUDA, and cuDNN** before any data loading occurs (`torch.manual_seed`, `np.random.seed`, `torch.cuda.manual_seed_all`, `torch.backends.cudnn.deterministic = True`, `torch.backends.cudnn.benchmark = False`)
- Strict cuDNN backend determinism is additionally enforced via the `CUBLAS_WORKSPACE_CONFIG` environment variable and `torch.use_deterministic_algorithms(True, warn_only=True)`, both set **before** any CUDA initialization
- Set the environment variable before importing torch, e.g.:
  ```bash
  export CUBLAS_WORKSPACE_CONFIG=:4096:8
  ```
- Each notebook's dataloader logs an MD5 hash of the source filepath alongside every tensor, so split membership can be independently verified against the values below.

**Dataset splits used:**
| Dataset | Split | Notes |
|---|---|---|
| DAGM | 80/20 train/validation | Frozen split (`GLOBAL_TRAIN_SPLIT = 0.8`, `GLOBAL_VAL_SPLIT = 0.2`); test set fixed per the DAGM standard benchmark definition |
| VisA | 60/20/20 train/validation/test | Deterministic split (`train_test_split`, `random_state=42`) applied strictly over the normal-image distribution; all anomalous images are routed to the test set; zero-overlap is explicitly asserted for all three split pairs (`train ∩ val`, `train ∩ test`, `val ∩ test`) at loader initialization |

---

## 3. Notebook → Paper Mapping

Every row below points to the exact notebook and cell range responsible for a specific table, figure, or reported number.

### Stage 1 — Spatial Localization (Table I, Table II Stage 1 rows, Table III Stage 1 rows, Fig. 2)

| Paper element | Dataset | Notebook | Cells |
|---|---|---|---|
| Table I — architecture selection ablation (ResNet-18 / Inception / Xception / PaDiM / PatchCore) | VisA | `VisA_NB0_Research.ipynb` | Cell 13 (Feature Extractor) → Cell 14 (PatchCore Config) → Cell 19 (Global Execution Loop & Ablation Study) → Cell 20 (Ablation Metrics Parsing) |
| Table I — architecture selection ablation | DAGM | `DAGM_NB0_Research.ipynb` | Cell 10 (PatchCore Config & Crop Extraction) → Cell 11 (Feature Extractor & Eval Loop) → Cell 14/14.1 (Metric Engine & Ablation Sweep) |
| Table II — Stage 1 rows (PCA dim, Gaussian σ ablation) | VisA | `VisA_NB0_Research.ipynb` | Cell 17.1 (Automated Ablation Grid Generator) → Cell 19 → Cell 20 |
| Table II — Stage 1 rows (PCA dim, Gaussian σ ablation) | DAGM | `DAGM_NB0_Research.ipynb` | Cell 13.1 (Automated Ablation Grid Generator) → Cell 14/14.1 |
| Table III — Stage 1 metrics (Image/Pixel AUROC, Pixel AP, Pixel AUPRO, train/inference time, peak VRAM) | VisA | `VisA_NB0_Research.ipynb` | Cell 19 → Cell 21 (Visualizing Ablation Tradeoffs) |
| Table III — Stage 1 metrics | DAGM | `DAGM_NB0_Research.ipynb` | Cell 14.2 (Ablation Logging & Global Summary) → Cell 16 (Handoff Verification) |
| Fig. 2 — PatchCore baseline learning phase | — | `VisA_NB0_Research.ipynb` or `DAGM_NB0_Research.ipynb` | Cells 13–17 (feature extraction → coreset subsampling → memory bank export) |
| K-Center Greedy coreset subsampling (ratio 0.1) | VisA | `VisA_NB0_Research.ipynb` | Cell 15 |
| K-Center Greedy coreset subsampling (ratio 0.1) | DAGM | `DAGM_NB0_Research.ipynb` | Cell 12 |

### Stage 2 — ROI Extraction (Table II Stage 2 rows, Table III Stage 2 rows, Fig. 3/4 crops)

| Paper element | Dataset | Notebook | Cells |
|---|---|---|---|
| Table II — Stage 2 rows (ROI coverage %, crop strategy, localization accuracy) | VisA | `VisA_NB1_Research.ipynb` | Cell 7 (Crop Extraction Function) → Cell 9.1 (Automated NB1 Ablation Grid Generator) → Cell 10 (Inference & Crop Extraction Loop) |
| Table II — Stage 2 rows | DAGM | `DAGM_NB1_Research.ipynb` | Cell 9 (Hybrid Threshold + Masking) → Cell 9.1 (Automated Ablation Grid Generator) → Cell 10 (Tight & Context Crop Extraction) |
| Table III — Stage 2 metrics (IoU, F1, False Positive Rate, processing overhead ms, peak VRAM) | VisA | `VisA_NB1_Research.ipynb` | Cell 16 (Global Metrics & Crop Overlap Aggregation) → Cell 17 (Final Packaging, Manifest Check & Unified Logging) |
| Table III — Stage 2 metrics | DAGM | `DAGM_NB1_Research.ipynb` | Cell 16 (Global Metrics & Ablation Results Table) → Cell 16.1 (Class-Wise Ablation Metrics) |
| Fig. 3/4 — tight crop / context crop visualization | VisA | `VisA_NB1_Research.ipynb` | Cell 11 (Visualizing the Anomaly Detection Pipeline) → Cell 12 (Final Output Verification, 6-Column Context View) |
| Fig. 3/4 — tight crop / context crop visualization | DAGM | `DAGM_NB1_Research.ipynb` | Cell 11.1 (Visual Ablation Crop Variance) → Cell 12 (Final Output Verification) |

### Stage 3 — Semantic Arbitration (Table II Stage 3 rows, Table III Stage 3 rows, Fig. 4 pipeline flow)

| Paper element | Dataset | Notebook | Cells |
|---|---|---|---|
| Table II — Stage 3 rows (TTA mode, entropy gate, crop fusion, final accuracy) | VisA | `VisA_NB2_Research.ipynb` | Cell 9 (Temperature Calibration, Entropy Threshold, Mahalanobis Fit) → Cell 10 (Deep Enhancement / TTA / Orthogonal Subspace Projection) → Cell 11 (Static Feature Fusion, Mahalanobis Gate) |
| Table II — Stage 3 rows | DAGM | `DAGM_NB2_Research.ipynb` | Cell 9 → Cell 10 → Cell 11 |
| Table III — Stage 3 metrics (Standard CLIP / AnomalyCLIP / Final Routed accuracy, win rates, end-to-end latency) | VisA | `VisA_NB2_Research.ipynb` | Cell 19 (Accuracy by Class) → Cell 23 (Quantitative Performance & Routing Verification Tables) |
| Table III — Stage 3 metrics | DAGM | `DAGM_NB2_Research.ipynb` | Cell 19 → Cell 23 (Quantitative Performance & Routing Verification Tables) |
| Fig. 4 — full end-to-end pipeline flow (one image per class) | VisA | `VisA_NB2_Research.ipynb` | Cell 17 (Paper Figure — Full Pipeline Flow) |
| Fig. 4 — full end-to-end pipeline flow | DAGM | `DAGM_NB2_Research.ipynb` | Cell 17 (Paper Figure — Full Pipeline Flow) |
| Shannon entropy arbitration landscape figure | VisA / DAGM | `NB2` (both) | Cell 16 (Single Image/Graph Visualization — Shannon Entropy Arbitration Landscape) |
| Crop-to-label routing verification (supports "roughly halved misclassification" claim) | VisA / DAGM | `NB2` (both) | Cell 18 (Crop-to-Label Provenance Table) |

---

## 4. Exact Hyperparameters

These are the values used to produce every reported number in the paper. Where a value differs by dataset, both are listed.

| Parameter | Value | Notes |
|---|---|---|
| Backbone | WideResNet-50, Layers 2 & 3 | PCA dimensionality reduction bypassed ("No PCA" configuration in Table II) |
| Coreset sampling ratio | 0.1 (K-Center Greedy) | |
| Gaussian smoothing kernel | σ = 0.1 | Reflect-padded, applied to the anomaly map before max-pooling |
| FAISS nearest-neighbor search | k = 9 | Exact search, `IndexFlatL2`, embeddings L2-normalized before indexing |
| Anomaly threshold (τ) | 99th percentile of nominal validation scores | Per-dataset, computed independently |
| ROI expansion (Context Crop) | **DAGM: 50%** · **VisA: 75%** | Per Table II ablation; ⚠️ see note below |
| Crop fusion weights | 0.7 · tight-crop vector + 0.3 · context-crop vector | Static linear fusion |
| Input resolution (Stage 1) | 512 × 512 | Standardized across both datasets |
| VisA-specific preprocessing | Automated background removal + Gaussian blur (kernel size 3) | Applied only to VisA to suppress high-frequency background noise |
| DAGM-specific preprocessing | Native background retained | To preserve continuous surface topology |
| VLM input resolution | 224 × 224 | ImageNet channel-statistic normalization applied before embedding |
| VLM 1 | Standard CLIP, ViT-B/32 | |
| VLM 2 | AnomalyCLIP, ViT-L/14 | |
| Temperature calibration | Per-model scalar T, fitted offline via negative log-likelihood minimization | Applied to raw logits before softmax |
| Entropy confusion threshold | 99th percentile of Shannon entropy H(P) | Above this, Test-Time Augmentation (TTA) is triggered |
| TTA augmented views | 4 | Embeddings averaged into V_TTA |
| Random seed | 42 | Frozen across PyTorch, NumPy, CUDA, cuDNN |

---

## 5. Regenerating the Memory Banks

The coreset memory banks (`memory_bank.pkl`) are not included in this repository due to file size; regenerate them as follows:

1. Set the deterministic environment variables **before** importing PyTorch:
   ```bash
   export CUBLAS_WORKSPACE_CONFIG=:4096:8
   ```
2. Open `<Dataset>_NB0_Research.ipynb` for the dataset you want (`VisA` or `DAGM`).
3. Run all cells through the **feature extractor and PatchCore configuration** cells (VisA: Cells 13–14; DAGM: Cells 10–11) with the hyperparameters in Section 4 above — confirm seed = 42 is set in Cell 1/2 before proceeding.
4. Run the **K-Center Greedy coreset subsampling** cell (VisA: Cell 15; DAGM: Cell 12) with sampling ratio = 0.1.
5. Run the **ablation grid / global execution loop** (VisA: Cells 17.1, 19; DAGM: Cells 13.1, 14/14.1) to reproduce Table I and the Stage 1 rows of Table II.
6. The memory bank is serialized to `.pkl` format at the end of the notebook — this is the artifact consumed by `NB1` in the next stage.
7. Proceed to `NB1` (ROI extraction) and then `NB2` (semantic arbitration) for the same dataset, in order, without changing the seed.

Each notebook logs an MD5 hash of every processed filepath — cross-check these hashes against your own run if you want to confirm you're operating on the identical data split used in the paper.

---

## 6. Reproducibility Notes

- All headline results in Tables I, II, and III are **single-run point estimates** (seed = 42); no variance across multiple seeds is currently reported.
- Peak VRAM, training time, and inference time figures are logged automatically by each notebook's GPU safety/logging utilities (see `NB0` Cell 19–21 and `NB2` Cell 6, 14) and reflect the Kaggle T4 GPU environment described in Section 2.
