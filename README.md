# 🔬 Melanoma Classification – MLDM2 Deep Learning Project

Deep Learning model to predict the probability that a skin lesion is malignant (melanoma), using image data combined with clinical metadata.

---

## 📌 Project Overview

| | |
|---|---|
| **Task** | Binary Classification (malignant / benign) |
| **Dataset** | [SIIM-ISIC Melanoma Classification 2020](https://www.kaggle.com/competitions/siim-isic-melanoma-classification) |
| **Data Type** | Multimodal: Images + Clinical Metadata |
| **Challenge** | Severe class imbalance (~2% positive cases) |
| **Compute** | LightningAI (GPU) |
| **Grade** | 20% of final module grade |

---

## 🗂️ Project Structure

```
MLDM2-MelanomaClassification/
├── data/                   # ⚠️ NOT in repo – download manually (see below)
│   ├── train/              # Training images (.png, 224x224)
│   ├── test/               # Test images (.png, 224x224)
│   ├── train.csv           # Labels & metadata (target, age, sex, body site)
│   └── test.csv            # Test metadata (no labels)
├── checkpoints/            # Saved model weights (auto-created during training)
├── tb_logs/                # TensorBoard logs (auto-created during training)
├── datamodule.py           # Data loading, augmentation, train/val/test split
├── model.py                # EfficientNet-B0 model architecture
├── train.py                # Training script – start here
├── eda.ipynb               # Exploratory data analysis notebook
├── requirements.txt        # Python dependencies
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

# If you get a PowerShell error run this first:
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
> **⚠️ Kaggle API Key required:** To use the CLI commands below, get your API key at [kaggle.com](https://www.kaggle.com/settings) → Settings → API → Create New Token. Place `kaggle.json` in `C:\Users\<yourname>\.kaggle\` (Windows) or `~/.kaggle/` (Mac/Linux).  

> Alternatively, download both sources manually from the Kaggle website and place them in `data/`.
The data comes from **two separate Kaggle sources**:

**Source 1 – Images** (224x224, ~3.5 GB) from [this dataset](https://www.kaggle.com/datasets/arroqc/siic-isic-224x224-images):
```bash
kaggle datasets download -d arroqc/siic-isic-224x224-images -p data/
cd data
tar -xf siic-isic-224x224-images.zip
cd ..
```
→ This creates `data/train/` and `data/test/` with `.png` images.

**Source 2 – CSV files** (labels & metadata) from the [Kaggle Competition](https://www.kaggle.com/competitions/siim-isic-melanoma-classification/data):
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

**Local test** (fast, only a few batches – use this to check everything works):
```bash
# In train.py: set "fast_dev": True
py train.py
```

**Full training** (on LightningAI with GPU):
```bash
# In train.py: set "fast_dev": False
python train.py
```

---

## ✅ Todo List

### 📁 Data & Preprocessing
- [x] Download dataset (images + CSVs)
- [x] Exploratory Data Analysis (EDA notebook)
- [ ] Train / Val / Test split (stratified)
- [ ] LightningDataModule implementation
- [ ] Data augmentation (flip, rotation, color jitter)

### ⚖️ Class Imbalance
- [ ] Choose strategy: WeightedRandomSampler / Weighted Loss / both
- [ ] Implement and evaluate chosen strategy

### 🧠 Model Architecture
- [ ] Baseline model (images only)
- [ ] Add metadata fusion (age, sex, body site)
- [ ] Define metrics: AUC-ROC, F1-Score, Confusion Matrix

### 🏋️ Training & Experiments
- [ ] Train baseline model
- [ ] Train multimodal model
- [ ] Tune hyperparameters (LR, batch size, backbone)
- [ ] Log experiments (TensorBoard or W&B)

### 📊 Evaluation & Analysis
- [ ] Evaluate on test set
- [ ] Compare baseline vs. multimodal
- [ ] Error analysis
- [ ] (Optional) Late submission on Kaggle leaderboard

### 📝 Report & Presentation
- [ ] Write PDF report (architecture, class imbalance, results)
- [ ] Prepare 10-minute presentation
- [ ] Deadline: **day before the last lecture**

---

## 📚 Resources

- [Kaggle Competition](https://www.kaggle.com/competitions/siim-isic-melanoma-classification)
- [Pre-processed 224x224 Images](https://www.kaggle.com/datasets/arroqc/siic-isic-224x224-images)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [timm – Pretrained Models](https://huggingface.co/docs/timm/index)

---

> **Note:** Existing Kaggle solutions may be used for inspiration, but the final solution must be your own.