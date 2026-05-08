# =============================================================================
# datamodule.py  –  Datenpipeline für Melanom-Klassifikation
#
# Was dieses File macht:
#   1. Lädt Bilder und CSV-Metadaten
#   2. Preprocesst Metadaten (Normalisierung, One-Hot Encoding)
#   3. Macht Train/Val/Test Split (stratifiziert → gleiches Klassenungleichgewicht)
#   4. Data Augmentation fürs Training (Bilder zufällig verändern → robusteres Modell)
#   5. WeightedRandomSampler → seltenere Melanom-Fälle häufiger samplen
# =============================================================================

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import lightning as L
from sklearn.model_selection import train_test_split


# =============================================================================
# DATASET KLASSE
# Gibt für jeden Index zurück: (Bild-Tensor, Metadaten-Tensor, Label)
# =============================================================================

class MelanomaDataset(Dataset):

    def __init__(self, df, image_dir, transform=None):
        """
        df         : DataFrame mit Spalten: image_name, target, meta_features
        image_dir  : Pfad zum Ordner mit .png Bildern
        transform  : Albumentations oder torchvision Augmentierungen
        """
        self.df        = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

        # Metadaten-Spalten als numpy array (vorberechnete Features aus preprocess_metadata)
        self.metadata = df[[c for c in df.columns if c.startswith("meta_")]].values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # ---- Bild laden & transformieren ----
        img_path = os.path.join(self.image_dir, row["image_name"] + ".jpg")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, row["image_name"] + ".png")

        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # ---- Metadaten ----
        metadata = torch.tensor(self.metadata[idx], dtype=torch.float32)

        # ---- Label ----
        label = torch.tensor(row["target"], dtype=torch.float32)

        return image, metadata, label


# =============================================================================
# METADATA PREPROCESSING
#
# Das Modell braucht Zahlen – keine Strings wie "male" oder "torso".
# Deshalb:
#   - Alter: Min-Max Normalisierung → Werte exakt zwischen 0 und 1
#            WICHTIG: min/max werden nur auf dem Trainingsset berechnet,
#            dann auf Val/Test angewendet → kein Data Leakage!
#   - Geschlecht: One-Hot Encoding  male=[1,0], female=[0,1]
#   - Körperstelle: One-Hot Encoding (6 häufigste Kategorien)
#   - Fehlende Werte: mit 0 füllen
#
# Resultat: meta_dim = 1 + 2 + 6 = 9 Zahlenwerte pro Patient
# =============================================================================

BODY_SITES = [
    "torso", "lower extremity", "upper extremity",
    "head/neck", "palms/soles", "oral/genital"
]

def preprocess_metadata(df, age_min=None, age_max=None):
    """
    Fügt meta_* Spalten zum DataFrame hinzu.

    age_min, age_max : Wenn None → werden aus df berechnet (nur für Trainingsset!)
                       Sonst → übergebene Werte verwenden (für Val/Test → kein Leakage)

    Gibt zurück: (DataFrame, meta_dim, age_min, age_max)
    """
    df = df.copy()

    # 1. Alter: Min-Max Normalisierung
    # Fehlende Werte zuerst mit 0 füllen, dann normalisieren
    age = df["age_approx"].fillna(0)

    if age_min is None:
        age_min = age.min()   # nur auf Trainingsset berechnen!
    if age_max is None:
        age_max = age.max()

    # Sicherheit: falls age_min == age_max (alle gleich alt) → Division durch 0 vermeiden
    if age_max > age_min:
        df["meta_age"] = (age - age_min) / (age_max - age_min)
    else:
        df["meta_age"] = 0.0

    # 2. Geschlecht → One-Hot
    df["meta_sex_male"]   = (df["sex"] == "male").astype(float)
    df["meta_sex_female"] = (df["sex"] == "female").astype(float)

    # 3. Körperstelle → One-Hot (6 Kategorien)
    for site in BODY_SITES:
        col = "meta_site_" + site.replace("/", "_").replace(" ", "_")
        df[col] = (df["anatom_site_general_challenge"] == site).astype(float)

    meta_cols = [c for c in df.columns if c.startswith("meta_")]
    meta_dim = len(meta_cols)  # = 9
    return df, meta_dim, age_min, age_max


# =============================================================================
# LIGHTNING DATA MODULE
# Kapselt den gesamten Datenpipeline in einer Klasse.
# Lightning ruft setup() und die *_dataloader() Methoden automatisch auf.
# =============================================================================

class MelanomaDataModule(L.LightningDataModule):

    def __init__(self, data_dir="data", batch_size=32, num_workers=4, val_size=0.15, test_size=0.15):
        super().__init__()
        self.data_dir    = data_dir
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.val_size    = val_size
        self.test_size   = test_size
        self.meta_dim    = None  # wird in setup() gesetzt

    def setup(self, stage=None):
        # ---- CSV laden ----
        df = pd.read_csv(os.path.join(self.data_dir, "train.csv"))

        # ---- Stratifizierter Split ZUERST, dann Preprocessing ----
        # Wichtig: Split VOR preprocess_metadata, damit age_min/max nur auf
        # Trainingsdaten berechnet werden → kein Data Leakage!

        # ---- Stratifizierter Split ----
        # Stratifiziert = gleiche Klassenverteilung in Train/Val/Test
        # Wichtig bei ~2% positiv – ohne Stratifizierung könnte Val kein Melanom haben!
        train_df, temp_df = train_test_split(
            df, test_size=self.val_size + self.test_size,
            stratify=df["target"], random_state=42
        )
        val_df, test_df = train_test_split(
            temp_df, test_size=self.test_size / (self.val_size + self.test_size),
            stratify=temp_df["target"], random_state=42
        )

        # ---- Metadaten preprocessen ----
        # age_min/max NUR auf train_df berechnen, dann auf val/test anwenden → kein Data Leakage!
        train_df, self.meta_dim, age_min, age_max = preprocess_metadata(train_df)
        val_df,   _,             _,       _        = preprocess_metadata(val_df,  age_min, age_max)
        test_df,  _,             _,       _        = preprocess_metadata(test_df, age_min, age_max)

        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        print(f"Train positiv: {train_df['target'].mean()*100:.2f}%")
        print(f"Age normalization: min={age_min:.0f}, max={age_max:.0f}")

        image_dir = os.path.join(self.data_dir, "train")

        # ---- Transformierungen ----
        # Training: Augmentation → Modell wird robuster (sieht jedes Bild leicht anders)
        # Val/Test:  nur normalisieren → faire Evaluation ohne Zufall
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],   # ImageNet Mittelwerte
                                 std=[0.229, 0.224, 0.225])     # (da Backbone darauf trainiert)
        ])
        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.train_dataset = MelanomaDataset(train_df, image_dir, transform=train_transform)
        self.val_dataset   = MelanomaDataset(val_df,   image_dir, transform=eval_transform)
        self.test_dataset  = MelanomaDataset(test_df,  image_dir, transform=eval_transform)

        # ---- WeightedRandomSampler (Class Imbalance) ----
        # Seltene Klasse (Melanom, ~2%) wird häufiger gesampelt → Modell sieht mehr Positive
        # Kombiniert mit pos_weight im Loss = doppelte Absicherung gegen Imbalance
        labels = train_df["target"].values
        class_counts  = np.bincount(labels)         # [Anzahl neg, Anzahl pos]
        class_weights = 1.0 / class_counts          # seltenere Klasse → höheres Gewicht
        sample_weights = class_weights[labels]      # jedem Sample sein Gewicht

        self.sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(sample_weights),
            replacement=True
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,       # WeightedRandomSampler statt shuffle=True
            num_workers=self.num_workers,
            pin_memory=True             # schnellerer Transfer CPU→GPU
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
