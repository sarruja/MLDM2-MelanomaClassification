# Melanoma Classification – MLDM2 Deep Learning Project

Deep learning model to predict the probability that a skin lesion is malignant (melanoma), using skin lesion images combined with clinical patient metadata.

---

## 📌 Project overview

| | |
|---|---|
| **Task** | Binary classification (malignant / benign) |
| **Dataset** | [SIIM-ISIC Melanoma Classification 2020](https://www.kaggle.com/competitions/siim-isic-melanoma-classification) |
| **Input** | Multimodal: skin lesion images + patient metadata |
| **Challenge** | Severe class imbalance (~2% positive cases) |
| **Compute** | LightningAI (GPU – T4) |
| **Grade** | 20% of final module grade |

---

## 📊 Results

### Multimodal vs. Baseline (Threshold = 0.50)

| Metric | Multimodal | Baseline (images only) |
|---|---|---|
| **AUC-ROC** | 0.9126 | 0.9090 |
| **F1-Score** | 0.2326 | 0.2374 |
| **Sensitivity** | 0.6322 | 0.6207 |
| **Specificity** | 0.9322 | 0.9357 |
| **TP** | 55 | 54 |
| **FN** | 32 | 33 |
| **FP** | 331 | 314 |
| Best epoch | 6 | 6 |

> AUC-ROC is the primary metric — accuracy is meaningless with ~2% positive cases.
> Multimodal model achieves higher AUC and Sensitivity; Baseline has slightly better Specificity and F1.
> The metadata changed the decision behaviour slightly, but image features still dominate.

### Threshold Analysis – Multimodal Model

| Threshold | Sensitivity | Specificity | F1-Score | TP | FN | FP |
|---|---|---|---|---|---|---|
| 0.05 | 0.782 | 0.894 | 0.202 | 68 | 19 | 517 |
| 0.10 | 0.736 | 0.907 | 0.212 | 64 | 23 | 452 |
| 0.20 | 0.736 | 0.917 | 0.229 | 64 | 23 | 407 |
| 0.30 | 0.701 | 0.922 | 0.230 | 61 | 26 | 382 |
| 0.50 | 0.632 | 0.932 | 0.233 | 55 | 32 | 331 |

> In a medical context, a lower threshold (e.g. 0.05–0.10) is preferable: missed melanomas (FN) are far more critical than false alarms (FP). At threshold 0.05, Sensitivity reaches 0.78 at the cost of more False Positives.

### pos_weight Hyperparameter Search

| pos_weight | AUC-ROC | Sensitivity | Specificity | TP | FN |
|---|---|---|---|---|---|
| 25 | 0.9124 | 0.402 | 0.977 | 35 | 52 |
| **50** | **0.9126** | **0.632** | **0.932** | **55** | **32** |
| 75 | 0.9182 | 0.598 | 0.939 | 52 | 35 |

> pos_weight=50 chosen as best trade-off between Sensitivity and AUC.

Full results (plots, metrics JSON, training history, error analysis) are saved in `results/`.

---

## 🧠 Model architecture

The model is multimodal: it processes the image and patient metadata separately, then fuses both into a single prediction.

```
Skin lesion image (224x224)          Patient metadata (age, sex, body site)
        |                                          |
EfficientNet-B0 Backbone                   Metadata MLP
(CNN, pretrained on ImageNet)          (Linear 9→64→32 + ReLU)
        |                                          |
 Image features [1280]               Metadata features [32]
        |                                          |
        +──────── Concatenation ───────────────────+
                        |
               Fused vector [1312]
                        |
              Classifier Head
       (Dropout → FC 1312→256 → ReLU → Dropout → FC 256→1)
                        |
              Output: probability (0–1)
              0.0 = benign  ·  1.0 = melanoma
```

**Key architecture decisions:**
- EfficientNet-B0 as image backbone — efficient CNN with residual connections, pretrained on ImageNet (transfer learning)
- Metadata MLP processes patient features separately before fusion — avoids drowning 9 metadata values in 1280 image features
- Concatenation fusion — simple and effective for combining heterogeneous inputs

---

## ⚖️ Handling class imbalance

With ~2% positive cases, a naive model reaches 98% accuracy by always predicting "benign" — which is useless. Two strategies are combined:

- **`pos_weight=50` in BCEWithLogitsLoss** — a missed melanoma is penalized ~50× more than a false alarm (derived from 98/2 ≈ 49 class ratio)
- **`WeightedRandomSampler`** — oversamples melanoma cases during training so the model sees them more often

Evaluation uses AUC-ROC and F1-score instead of accuracy.

---

## 🧬 Patient metadata features (9 total)

| Feature | Encoding | Dim |
|---|---|---|
| Age | Min-max normalization fitted on training set (no data leakage) | 1 |
| Sex | One-hot (male / female) | 2 |
| Body site | One-hot (torso / lower extremity / upper extremity / head-neck / palms-soles / oral-genital) | 6 |

---

## 🗂️ Project structure

```
MLDM2-MelanomaClassification/
├── data/                        # NOT in repo – download manually (see below)
│   ├── train/                   # 33126 training images (.png, 224x224)
│   ├── test/                    # 10982 test images (.png, 224x224)
│   ├── train.csv                # Labels & metadata (target, age, sex, body site)
│   └── test.csv                 # Test metadata (no labels)
├── checkpoints/                 # Saved model weights (auto-created, not in repo)
├── tb_logs/                     # TensorBoard logs (auto-created, not in repo)
├── results/                     # Evaluation outputs (in repo)
│   ├── multimodal/              # metrics.json, roc_curve.png, confusion_matrix.png,
│   │                            # training_history.png, error_analysis.png,
│   │                            # threshold_comparison.csv
│   ├── baseline/                # same as multimodal/
│   ├── hparam_pos_weight_25/    # metrics.json
│   ├── hparam_pos_weight_50/    # metrics.json
│   ├── hparam_pos_weight_75/    # metrics.json
│   └── hparam_comparison.csv   # pos_weight experiment comparison
├── datamodule.py                # Data loading, preprocessing, augmentation, splits
├── model.py                     # Multimodal EfficientNet-B0 model architecture
├── model_baseline.py            # Baseline model (images only, no metadata)
├── model_v2.py                  # V2 model (improved fusion architecture)
├── train.py                     # Multimodal training script
├── train_baseline.py            # Baseline training script
├── train_v2.py                  # V2 training script
├── hyperparameter_search.py     # pos_weight hyperparameter experiments
├── evaluate.py                  # Evaluation, threshold analysis & plots
├── eda.ipynb                    # Exploratory data analysis notebook
├── requirements.txt             # Python dependencies
├── .gitignore
└── README.md
```

---

## 📦 Local setup

### 1. Clone the repo
```bash
git clone https://github.com/sarruja/MLDM2-MelanomaClassification.git
cd MLDM2-MelanomaClassification
```

### 2. Create & activate virtual environment

**Windows:**
```bash
py -m venv melanoma-env

# If you get a PowerShell error, run this first:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

melanoma-env\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv melanoma-env
source melanoma-env/bin/activate
```

You should see `(melanoma-env)` in front of your prompt. ✅

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download the data

> **Kaggle API key required:** get it at [kaggle.com](https://www.kaggle.com/settings) → Settings → API → Create New Token. Place `kaggle.json` in `C:\Users\<yourname>\.kaggle\` (Windows) or `~/.kaggle/` (Mac/Linux). Alternatively download manually from the Kaggle website.

**Source 1 – Images** (~3.5 GB, 33126 train / 10982 test):
```bash
kaggle datasets download -d arroqc/siic-isic-224x224-images -p data/
cd data && unzip siic-isic-224x224-images.zip && cd ..
```

**Source 2 – CSV files** (labels & metadata):
```bash
kaggle competitions download -c siim-isic-melanoma-classification -f train.csv -p data/
kaggle competitions download -c siim-isic-melanoma-classification -f test.csv -p data/
```

> If the competition download gives a 401 error, accept the competition rules at [kaggle.com/competitions/siim-isic-melanoma-classification/data](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data) first. If it still fails, upload `train.csv` and `test.csv` manually into `data/`.

---

## ☁️ LightningAI setup (GPU training)

### 1. Open the team workspace
Go to [lightning.ai](https://lightning.ai) → switch to the `DLFS2026_1` teamspace.

### 2. Create a new Studio
New Studio → **AI development** → Machine: **1 × T4** → Start.

> **Tip:** Click the machine indicator (top right) → enable **Auto sleep off** so the studio doesn't pause during training.

### 3. Clone the repo & install dependencies
```bash
git clone https://github.com/sarruja/MLDM2-MelanomaClassification.git
cd MLDM2-MelanomaClassification
pip install -r requirements.txt
```

### 4. Set up Kaggle API key
```bash
mkdir -p ~/.kaggle
echo '{"username":"YOUR_USERNAME","key":"YOUR_KEY"}' > ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json
```

### 5. Download the data
```bash
kaggle datasets download -d arroqc/siic-isic-224x224-images -p data/
cd data && unzip siic-isic-224x224-images.zip && cd ..
```
Upload `train.csv` and `test.csv` manually into `data/` via drag & drop.

---

## 🚀 Running the code

**Local test** (fast, only 1 batch):
```bash
# In train.py: set "fast_dev": True
py train.py
```

**Full training on LightningAI:**
```bash
# Multimodal model
python train.py

# Baseline model
python train_baseline.py

# Hyperparameter search (pos_weight)
python hyperparameter_search.py
```

**Evaluate & save results:**
```bash
# Multimodal
python evaluate.py --model multimodal --checkpoint checkpoints/hparam_pos_weight_50/pos50-best-epoch=06-val_auc=0.9044.ckpt

# Baseline
python evaluate.py --model baseline --checkpoint checkpoints/baseline/baseline-best-epoch=06-val_auc=0.8921.ckpt
```

Results are saved to `results/multimodal/` and `results/baseline/`, including:
- `metrics.json` – all metrics for all thresholds
- `threshold_comparison.csv` – threshold analysis table
- `error_analysis.png` – FP / FN image grid
- `roc_curve.png`, `confusion_matrix.png`, `precision_recall_curve.png`
- `training_history.png` / `training_history.csv`

**Resume training after interruption:**
```bash
# In train.py: uncomment resume_from = "checkpoints/last.ckpt"
python train.py
```

---

## ✅ Todo

### 📁 Data & preprocessing
- [x] Download dataset (images + CSVs)
- [x] Exploratory Data Analysis (EDA notebook)
- [x] Stratified train/val/test split (70/15/15)
- [x] Metadata preprocessing (min-max age, one-hot sex & body site)
- [x] Data augmentation (flip, rotation, color jitter)

### 🧠 Model
- [x] EfficientNet-B0 backbone (transfer learning)
- [x] Metadata MLP + concatenation fusion
- [x] BCEWithLogitsLoss with pos_weight=50
- [x] WeightedRandomSampler
- [x] Metrics: AUC-ROC, F1-score
- [x] Baseline model (images only) for comparison

### 🏋️ Training & experiments
- [x] Full training run on LightningAI (GPU)
- [x] Compare baseline vs. multimodal model
- [x] Log experiments (TensorBoard)
- [x] Hyperparameter search (pos_weight: 25 / 50 / 75)

### 📊 Evaluation
- [x] AUC-ROC, F1-score, confusion matrix on test set
- [x] Training history (loss/auc per epoch)
- [x] Threshold analysis (0.05 / 0.10 / 0.20 / 0.30 / 0.50)
- [x] Error analysis (FP / FN image grid)
- [ ] (Optional) Late submission on Kaggle leaderboard

### 📝 Report & presentation
- [ ] PDF report (architecture decisions, class imbalance handling, results analysis)
- [ ] 10-minute presentation (all team members present on site)
- [ ] Deadline: day before last lecture

---

## 📚 Resources

- [Kaggle Competition](https://www.kaggle.com/competitions/siim-isic-melanoma-classification)
- [Pre-processed 224x224 Images](https://www.kaggle.com/datasets/arroqc/siic-isic-224x224-images)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [timm – Pretrained Models](https://huggingface.co/docs/timm/index)
