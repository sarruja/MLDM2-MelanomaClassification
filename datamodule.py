# =============================================================================
# datamodule.py
# Zuständig für: Daten laden, aufteilen, augmentieren und als Batches liefern
# Lesson 1: Train/Val/Test Split
# Lesson 3: Data Augmentation
# =============================================================================

import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import WeightedRandomSampler
import torchvision.transforms as transforms

import lightning as L
from sklearn.model_selection import train_test_split


# =============================================================================
# DATASET KLASSE
# Definiert wie ein einzelnes Datensample aussieht (Bild + Label)
# PyTorch erwartet eine __len__() und __getitem__() Methode
# =============================================================================

class MelanomaDataset(Dataset):
    def __init__(self, df, image_dir, transform=None):
        """
        df         : Pandas DataFrame mit Spalten 'image_name' und 'target'
        image_dir  : Pfad zum Ordner mit den Bildern
        transform  : Bildtransformationen (Augmentation oder nur Normalisierung)
        """
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        # Gibt die Anzahl der Samples zurück
        return len(self.df)

    def __getitem__(self, idx):
        # Lädt ein einzelnes Sample anhand des Index
        row = self.df.iloc[idx]

        # Bild laden
        img_path = os.path.join(self.image_dir, row["image_name"] + ".png")
        image = Image.open(img_path).convert("RGB")

        # Transformationen anwenden (z.B. Augmentation beim Training)
        if self.transform:
            image = self.transform(image)

        # Label: 1 = Melanom (bösartig), 0 = gutartig
        label = torch.tensor(row["target"], dtype=torch.float32)

        return image, label


# =============================================================================
# DATAMODULE KLASSE (Lightning)
# Kapselt die gesamte Datenpipeline in einer Klasse
# Lightning ruft setup() und die *_dataloader() Methoden automatisch auf
# =============================================================================

class MelanomaDataModule(L.LightningDataModule):
    def __init__(self, csv_path, image_dir, batch_size=32, val_split=0.15, test_split=0.15, num_workers=4):
        """
        csv_path   : Pfad zur train.csv (enthält image_name, target, Metadaten)
        image_dir  : Pfad zum Bildordner
        batch_size : Anzahl Bilder pro Batch (klein = weniger GPU-RAM, langsamer)
        val_split  : Anteil Daten für Validation (z.B. 0.15 = 15%)
        test_split : Anteil Daten für Test
        num_workers: Parallele Prozesse zum Laden der Bilder (0 = kein Parallelismus)
        """
        super().__init__()
        self.csv_path = csv_path
        self.image_dir = image_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.num_workers = num_workers

        # -------------------------------------------------------
        # TRANSFORMS / AUGMENTATION (Lesson 3 - Data Augmentation)
        # Training: Zufällige Transformationen → Modell wird robuster
        # Validation/Test: NUR Normalisierung → faire Evaluation
        # -------------------------------------------------------

        # Für Training: Augmentation aktiviert
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),           # Bilder auf 224x224 bringen (unser Dataset ist schon so)
            transforms.RandomHorizontalFlip(),        # Zufällig horizontal spiegeln
            transforms.RandomVerticalFlip(),          # Zufällig vertikal spiegeln
            transforms.RandomRotation(degrees=15),   # Zufällig bis 15 Grad drehen
            transforms.ColorJitter(                  # Farbe leicht variieren
                brightness=0.2,
                contrast=0.2,
                saturation=0.2
            ),
            transforms.ToTensor(),                   # PIL Image → Tensor (Werte 0-1)
            transforms.Normalize(                    # Normalisierung auf ImageNet-Werte
                mean=[0.485, 0.456, 0.406],          # (weil EfficientNet auf ImageNet trainiert wurde)
                std=[0.229, 0.224, 0.225]
            ),
        ])

        # Für Validation & Test: Keine Augmentation, nur normalisieren
        self.val_test_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def setup(self, stage=None):
        """
        Wird von Lightning automatisch aufgerufen bevor das Training startet.
        Hier: CSV laden, aufteilen in Train/Val/Test
        """
        # CSV mit allen Labels und Metadaten laden
        df = pd.read_csv(self.csv_path)

        # -------------------------------------------------------
        # TRAIN / VAL / TEST SPLIT (Lesson 1 - Train/Val/Test Split)
        # Stratified = gleichmässige Klassenverteilung in allen Splits
        # Wichtig bei Class Imbalance! (~2% positiv)
        # -------------------------------------------------------
        train_df, test_df = train_test_split(
            df,
            test_size=self.test_split,
            stratify=df["target"],   # gleiche Klassenverteilung sicherstellen
            random_state=42
        )
        train_df, val_df = train_test_split(
            train_df,
            test_size=self.val_split / (1 - self.test_split),
            stratify=train_df["target"],
            random_state=42
        )

        print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
        print(f"Train positiv: {train_df['target'].mean():.2%}")  # zeigt wie unbalanciert

        # Dataset-Objekte erstellen
        self.train_dataset = MelanomaDataset(train_df, self.image_dir, self.train_transform)
        self.val_dataset   = MelanomaDataset(val_df,   self.image_dir, self.val_test_transform)
        self.test_dataset  = MelanomaDataset(test_df,  self.image_dir, self.val_test_transform)

        # -------------------------------------------------------
        # CLASS IMBALANCE: WeightedRandomSampler
        # Problem: ~98% negativ, ~2% positiv → Modell lernt nur "alles negativ"
        # Lösung: Positive Samples öfter samplen beim Training
        # -------------------------------------------------------
        labels = train_df["target"].values
        class_counts = np.bincount(labels)                    # [Anzahl negativ, Anzahl positiv]
        class_weights = 1.0 / class_counts                   # Seltene Klasse → höheres Gewicht
        sample_weights = class_weights[labels]               # Jedes Sample bekommt sein Gewicht

        self.sampler = WeightedRandomSampler(
            weights=torch.tensor(sample_weights, dtype=torch.float32),
            num_samples=len(sample_weights),
            replacement=True   # Sampling mit Zurücklegen (nötig für Oversampling)
        )

    def train_dataloader(self):
        # Training: WeightedRandomSampler verwenden (gegen Class Imbalance)
        # shuffle=False weil der Sampler das Mischen übernimmt
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            sampler=self.sampler,
            num_workers=self.num_workers,
            pin_memory=True   # schnellerer Transfer CPU → GPU
        )

    def val_dataloader(self):
        # Validation: kein Sampler, kein Shuffle → faire Evaluation
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )

    def test_dataloader(self):
        # Test: genau wie Validation
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
        )
