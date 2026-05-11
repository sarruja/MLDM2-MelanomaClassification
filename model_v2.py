# =============================================================================
# model_v2.py  –  Multimodales Melanom-Modell mit verbesserter Fusion
#
# Verbesserung gegenüber model.py (v1):
#   v1: Bild (1280) + Metadaten (32) → concat [1312] → Classifier
#       Problem: 1280 >> 32 → Metadaten werden "begraben"
#
#   v2: Bild (1280) → FC → [128]
#       Metadaten (9) → FC → [128]
#       concat [256] → Classifier
#       Vorteil: Beide Modalitäten auf gleicher Dimension → gleiches Gewicht
#
# Relevante Lektionen:
#   Lesson 2 – Feedforward Networks (FC-Layer)
#   Lesson 3 – Transfer Learning, Dropout, Regularization
#   Lesson 4 – CNNs (EfficientNet ist ein CNN)
#   Lesson 5 – Residual Networks (EfficientNet baut auf ResNet-Ideen auf)
# =============================================================================

import torch
import torch.nn as nn
import lightning as L
import timm

from torchmetrics import AUROC, F1Score


# =============================================================================
# ARCHITEKTUR-ÜBERSICHT (Verbesserte Multimodal Fusion v2)
#
#   Bild (224x224x3)
#       │
#   EfficientNet-B0 Backbone (CNN, vortrainiert auf ImageNet)
#       │
#   Global Average Pooling
#       │
#   Image Feature Vector (1280-dim)
#       │
#   Image Projection FC  (1280 → 128)    Metadaten (Alter, Geschlecht, Körperstelle)
#       │                                    │
#   Image Features [128]               Metadata MLP (9 → 64 → 128)
#       │                                    │
#       │                            Metadata Features [128]
#       │                                    │
#       └──────────── Konkatenation ─────────┘
#                           │
#                   [128 + 128] = 256-dim
#                           │
#                   Classifier Head (FC-Layer)
#                           │
#                   Output: 1 Wert (logit → Wahrscheinlichkeit Melanom)
#
# Warum besser als v1?
#   v1: [1280 + 32] → Metadaten haben nur 2.4% des Vektors → werden ignoriert
#   v2: [128  + 128] → beide Modalitäten gleichwertig → Metadaten können beitragen
# =============================================================================


class MelanomaModelV2(L.LightningModule):

    def __init__(self, metadata_dim=9, proj_dim=128, learning_rate=1e-4, pos_weight=50.0, dropout=0.3):
        """
        metadata_dim  : Anzahl Input-Features der Metadaten (default: 9)
        proj_dim      : Projektionsdimension für Bild UND Metadaten (default: 128)
                        Beide werden auf proj_dim projiziert → gleiche Dimension → gleiches Gewicht
        learning_rate : Lernrate für Adam Optimizer
        pos_weight    : Gewicht für positive Klasse im Loss (~98/2 ≈ 50)
        dropout       : Dropout-Rate gegen Overfitting (Lesson 3)
        """
        super().__init__()
        self.save_hyperparameters()  # speichert alle Parameter für Checkpoints

        # ------------------------------------------------------------------
        # 1. BACKBONE: EfficientNet-B0 (Transfer Learning)
        #
        # EfficientNet wurde auf ImageNet vortrainiert (1.2 Mio Bilder).
        # Es hat bereits gelernt: Kanten, Texturen, Formen, komplexe Muster.
        # pretrained=True  → lädt diese Gewichte herunter (nicht selbst trainieren!)
        # num_classes=0    → entfernt den originalen Classifier-Head (1000 ImageNet-Klassen)
        # global_pool="avg"→ macht aus dem letzten Feature-Map einen Vektor (Average Pooling)
        # ------------------------------------------------------------------
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
            global_pool="avg"
        )
        backbone_out = self.backbone.num_features  # = 1280 bei EfficientNet-B0

        # ------------------------------------------------------------------
        # 2. IMAGE PROJECTION (neu in v2!)
        #
        # Reduziert die 1280 Bild-Features auf proj_dim (128).
        # Ziel: gleiche Dimension wie Metadaten-Features → gleiches Gewicht bei Fusion.
        # Das ist ein einfacher FC-Layer (Lesson 2 – Feedforward Networks).
        # ------------------------------------------------------------------
        self.image_proj = nn.Sequential(
            nn.Linear(backbone_out, proj_dim),  # 1280 → 128
            nn.ReLU(),
            nn.Dropout(p=dropout)
        )

        # ------------------------------------------------------------------
        # 3. METADATA MLP (erweitert in v2)
        #
        # Neu: Output ist proj_dim (128) statt 32 → gleiche Dimension wie Bild
        # ------------------------------------------------------------------
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 64),    # 9 → 64
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, proj_dim),        # 64 → 128
            nn.ReLU()
        )

        # ------------------------------------------------------------------
        # 4. FUSION CLASSIFIER HEAD
        #
        # v1: [1280 + 32]  = 1312-dim (Metadaten hatten nur 2.4% Anteil)
        # v2: [128  + 128] =  256-dim (beide Modalitäten gleichwertig: 50%/50%)
        # ------------------------------------------------------------------
        fusion_in = proj_dim + proj_dim  # 128 + 128 = 256

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fusion_in, 128),     # 256 → 128
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(128, 1)              # 128 → 1 (Logit, kein Sigmoid!)
        )

        # ------------------------------------------------------------------
        # 4. LOSS-FUNKTION: BCEWithLogitsLoss mit pos_weight
        #
        # BCE = Binary Cross Entropy (standard für binäre Klassifikation)
        # WithLogits = erwartet Logits (nicht Sigmoid-Output) → stabiler
        # pos_weight = 50 → ein falsch klassifiziertes Melanom wird 50x stärker
        #                    bestraft als ein falsch klassifiziertes Nicht-Melanom
        #
        # Ohne pos_weight: Modell lernt "sag immer 0" → 98% Accuracy aber nutzlos!
        # ------------------------------------------------------------------
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

        # ------------------------------------------------------------------
        # 5. METRIKEN
        #
        # Accuracy allein ist bei Klassenungleichgewicht wertlos (98% ohne Training!).
        # Deshalb:
        #   AUC-ROC: Wie gut trennt das Modell positiv/negativ bei allen Schwellwerten?
        #            1.0 = perfekt, 0.5 = zufällig
        #   F1-Score: Harmonisches Mittel aus Precision & Recall
        #             Gut wenn sowohl False Positives als auch False Negatives wichtig sind
        # ------------------------------------------------------------------
        self.val_auc = AUROC(task="binary")
        self.val_f1  = F1Score(task="binary")
        self.test_auc = AUROC(task="binary")
        self.test_f1  = F1Score(task="binary")

    # ------------------------------------------------------------------
    # FORWARD PASS
    # Wird bei jedem Schritt aufgerufen: Eingabe → Ausgabe
    # images   : Tensor [batch_size, 3, 224, 224]
    # metadata : Tensor [batch_size, metadata_dim]
    # ------------------------------------------------------------------
    def forward(self, images, metadata):
        # Bild durch EfficientNet → Feature-Vektor [batch, 1280]
        image_features = self.backbone(images)

        # Bild-Features projizieren: [batch, 1280] → [batch, 128]  (neu in v2!)
        image_proj = self.image_proj(image_features)

        # Metadaten durch MLP → Feature-Vektor [batch, 128]
        meta_features = self.metadata_mlp(metadata)

        # Beide Vektoren nebeneinander hängen → [batch, 256]
        # v1 war: [batch, 1312] — Metadaten hatten nur 2.4% Anteil
        # v2 ist: [batch, 256]  — beide 50%/50% gleichwertig
        combined = torch.cat([image_proj, meta_features], dim=1)

        # Durch Classifier → [batch, 1]
        logits = self.classifier(combined)
        return logits.squeeze(1)

    # ------------------------------------------------------------------
    # TRAINING STEP
    # Lightning ruft das automatisch für jeden Batch auf
    # ------------------------------------------------------------------
    def training_step(self, batch, batch_idx):
        images, metadata, labels = batch
        logits = self(images, metadata)
        loss = self.criterion(logits, labels.float())
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # VALIDATION STEP
    # Nach jeder Epoche auf dem Validierungsset
    # ------------------------------------------------------------------
    def validation_step(self, batch, batch_idx):
        images, metadata, labels = batch
        logits = self(images, metadata)
        loss = self.criterion(logits, labels.float())

        # Sigmoid: Logits → Wahrscheinlichkeiten (0–1) für Metriken
        probs = torch.sigmoid(logits)

        self.val_auc.update(probs, labels)
        self.val_f1.update(probs, labels)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        auc = self.val_auc.compute()
        f1  = self.val_f1.compute()
        self.log("val_auc", auc, prog_bar=True)
        self.log("val_f1",  f1,  prog_bar=True)
        self.val_auc.reset()
        self.val_f1.reset()

    # ------------------------------------------------------------------
    # TEST STEP
    # Einmal am Ende auf dem Test-Set (nie während dem Training anschauen!)
    # ------------------------------------------------------------------
    def test_step(self, batch, batch_idx):
        images, metadata, labels = batch
        logits = self(images, metadata)
        probs = torch.sigmoid(logits)
        self.test_auc.update(probs, labels)
        self.test_f1.update(probs, labels)

    def on_test_epoch_end(self):
        auc = self.test_auc.compute()
        f1  = self.test_f1.compute()
        self.log("test_auc", auc)
        self.log("test_f1",  f1)
        print(f"\n✅ Test AUC-ROC : {auc:.4f}")
        print(f"✅ Test F1-Score : {f1:.4f}")
        self.test_auc.reset()
        self.test_f1.reset()

    # ------------------------------------------------------------------
    # OPTIMIZER & LR SCHEDULER
    # Adam ist der Standard-Optimizer für Deep Learning
    # ReduceLROnPlateau: halbiert die LR wenn val_loss nicht besser wird
    # → hilft dem Modell am Ende feiner zu optimieren
    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",       # val_loss soll sinken
            factor=0.5,       # LR × 0.5 wenn kein Fortschritt
            patience=3        # warte 3 Epochen bevor LR gesenkt wird
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"}
        }
