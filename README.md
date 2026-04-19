# 🔬 Melanoma Classification – MLDM2 Deep Learning Project

Dieses Projekt ist Teil des Moduls **MLDM2** und hat zum Ziel, ein Deep Learning Modell zu entwickeln, das anhand von Hautläsions-Bildern und klinischen Metadaten die Wahrscheinlichkeit vorhersagt, ob ein Melanom bösartig ist.

---

## 📌 Projektübersicht

| | |
|---|---|
| **Aufgabe** | Binäre Klassifikation (bösartig / gutartig) |
| **Dataset** | [SIIM-ISIC Melanoma Classification 2020](https://www.kaggle.com/competitions/siim-isic-melanoma-classification) |
| **Datentyp** | Multimodal: Bilder + klinische Metadaten |
| **Herausforderung** | Starkes Class Imbalance (~2% positive Fälle) |
| **Compute** | LightningAI (GPU) |
| **Bewertung** | 20% der Modulnote |

---

## 🗂️ Projektstruktur

```
MLDM2-MelanomaClassification/
├── data/               # Daten (nicht ins Repo pushen!)
│   ├── images/         # 224x224 vorverarbeitete Bilder
│   └── train.csv       # Labels & Metadaten
├── checkpoints/        # Gespeicherte Modellgewichte
├── tb_logs/            # TensorBoard Logs
├── reports/            # Abschlussbericht (PDF)
├── datamodule.py       # LightningDataModule (Daten laden & augmentieren)
├── model.py            # LightningModule (EfficientNet Architektur)
├── train.py            # Training starten
├── requirements.txt    # Package-Versionen
├── .gitignore
└── README.md
```

---

## 📦 Setup

### 1. Repo klonen
```bash
git clone https://github.com/sarruja/MLDM2-MelanomaClassification.git
cd MLDM2-MelanomaClassification
```

### 2. Virtual Environment erstellen & aktivieren

**Windows:**
```bash
py -m venv melanoma-env

# Falls Fehler mit PowerShell (nur Windows):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

melanoma-env\Scripts\activate
```

**Mac/Linux:**
```bash
python3 -m venv melanoma-env
source melanoma-env/bin/activate
```

Wenn aktiviert siehst du `(melanoma-env)` vor dem Prompt. ✅

### 3. Dependencies installieren
```bash
pip install -r requirements.txt

```
oder pip install lightning timm torchmetrics albumentations pandas scikit-learn wenn no

### 4. Daten herunterladen (Kaggle API benötigt)
```bash
# kaggle.json API-Key unter: kaggle.com → Settings → API → Create New Token
# Key ablegen unter: C:\Users\<name>\.kaggle\kaggle.json

kaggle datasets download -d arroqc/siic-isic-224x224-images -p data/
```

---

## 🚀 Training starten

**Lokal testen** (schnell, nur wenige Batches):
```bash
# In train.py: "fast_dev": True setzen
py train.py
```

**Echtes Training** (auf LightningAI mit GPU):
```bash
# In train.py: "fast_dev": False setzen
python train.py
```

---

## ✅ Todo-Liste

### 📁 Daten & Preprocessing
- [ ] Kaggle-Dataset herunterladen (224x224 Version, ~3.5 GB)
- [ ] Explorative Datenanalyse (EDA): Klassenverteilung, Metadaten analysieren
- [ ] Daten-Split: Train / Validation / Test (stratifiziert!)
- [ ] `LightningDataModule` implementieren
- [ ] Data Augmentation definieren (Flip, Rotation, Color Jitter, etc.)

### ⚖️ Class Imbalance
- [ ] Strategie auswählen: Oversampling / Weighted Loss / beide?
- [ ] `WeightedRandomSampler` oder `pos_weight` in Loss implementieren
- [ ] Effekt der Strategie evaluieren

### 🧠 Modellarchitektur
- [ ] Backbone auswählen (z.B. EfficientNet, ResNet, ViT via `timm`)
- [ ] Multimodale Fusion: Metadaten (Alter, Geschlecht) mit Bildfeatures kombinieren
- [ ] `LightningModule` mit Forward Pass, Loss, Optimizer implementieren
- [ ] Metriken definieren: AUC-ROC, F1-Score, Confusion Matrix

### 🏋️ Training & Experimente
- [ ] Erstes Baseline-Modell trainieren (nur Bilder)
- [ ] Multimodales Modell trainieren (Bilder + Metadaten)
- [ ] Hyperparameter tunen (LR, Batch Size, Backbone)
- [ ] Early Stopping & Model Checkpointing einrichten
- [ ] Experimente loggen (W&B oder TensorBoard)

### 📊 Evaluation & Analyse
- [ ] Modelle auf dem Test-Set evaluieren
- [ ] Ergebnisse vergleichen (Baseline vs. multimodal)
- [ ] Fehleranalyse: Welche Bilder werden falsch klassifiziert?
- [ ] (Optional) Late Submission auf Kaggle für Leaderboard-Vergleich

### 📝 Bericht & Präsentation
- [ ] Bericht schreiben (PDF):
  - [ ] Architekturentscheidungen begründen
  - [ ] Umgang mit Class Imbalance erklären
  - [ ] Multimodale Integration beschreiben
  - [ ] Ergebnisse analysieren & interpretieren
- [ ] Präsentation vorbereiten (10 Minuten, alle Teammitglieder anwesend)
- [ ] Abgabe: **Tag vor der letzten Vorlesung**

---

## 📚 Ressourcen

- [Kaggle Competition](https://www.kaggle.com/competitions/siim-isic-melanoma-classification)
- [Pre-processed 224x224 Images](https://www.kaggle.com/datasets/arroqc/siic-isic-224x224-images)
- [PyTorch Lightning Docs](https://lightning.ai/docs/pytorch/stable/)
- [timm – Pretrained Image Models](https://huggingface.co/docs/timm/index)

---


> **Hinweis:** Originallösungen von Kaggle können zur Inspiration genutzt werden, die finale Lösung muss aber eigenständig sein.