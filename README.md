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
| **Compute** | LightningAI (GPU) |
| **Grade** | 20% of final module grade |

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
├── data/               # NOT in repo – download manually (see below)
│   ├── train/          # Training images (.png, 224x224)
│   ├── test/           # Test images (.png, 224x224)
│   ├── train.csv       # Labels & metadata (target, age, sex, body site)
│   └── test.csv        # Test metadata (no labels)
├── checkpoints/        # Saved model weights (auto-created during training)
├── tb_logs/            # TensorBoard logs (auto-created during training)
├── datamodule.py       # Data loading, preprocessing, augmentation, splits
├── model.py            # Multimodal EfficientNet-B0 model architecture
├── train.py            # Training script – start here
├── eda.ipynb           # Exploratory data analysis notebook
├── requirements.txt    # Python dependencies
├── .gitignore
└── README.md
```

---

## 📦 Setup

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

> **Kaggle API key required:** get it at [kaggle.com](https://www.kaggle.com/settings) → Settings → API → Create New Token. Place `kaggle.json` in `C:\Users\<yourname>\.kaggle\` (Windows) or `~/.kaggle/` (Mac/Linux). Alternatively download both sources manually from the Kaggle website.

**Source 1 – Images** (224x224, ~3.5 GB):
```bash
kaggle datasets download -d arroqc/siic-isic-224x224-images -p data/
cd data
tar -xf siic-isic-224x224-images.zip
cd ..
```

**Source 2 – CSV files** (labels & metadata):
```bash
kaggle competitions download -c siim-isic-melanoma-classification -f train.csv -p data/
kaggle competitions download -c siim-isic-melanoma-classification -f test.csv -p data/
```

Your `data/` folder should look like this:
```
data/
├── train/
│   ├── ISIC_0015719.png
│   └── ...
├── test/
│   └── ...
├── train.csv
└── test.csv
```

---

## 🚀 Running the code

**EDA first** – check data and visualize class distribution:
```bash
# Open eda.ipynb in VS Code and run all cells
```

**Local test** (fast, only 1 batch – verify everything works):
```bash
# In train.py: set "fast_dev": True
py train.py
```

**Full training** (on LightningAI with GPU):
```bash
# In train.py: set "fast_dev": False
python train.py
```

Lightning automatically detects and uses GPU if available (`accelerator="auto"`).

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
- [ ] Baseline model (images only) for comparison

### 🏋️ Training & experiments
- [ ] Full training run on LightningAI (GPU)
- [ ] Compare baseline vs. multimodal model
- [ ] Tune hyperparameters (LR, batch size, pos_weight, dropout)
- [ ] Log experiments (TensorBoard)

### 📊 Evaluation
- [ ] AUC-ROC, F1-score, confusion matrix on test set
- [ ] Error analysis (which images are misclassified?)
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

---

> Existing Kaggle solutions may be used for inspiration, but the final solution must be your own.