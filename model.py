# =============================================================================
# model.py  –  Multimodales Melanom-Klassifikationsmodell
#
# Was dieses File macht:
#   1. Lädt EfficientNet-B0 als vortrainiertes CNN (Transfer Learning)
#   2. Verarbeitet Patientenmetadaten (Alter, Geschlecht, Körperstelle) separat
#   3. Kombiniert (fusioniert) Bild-Features + Metadaten → finale Vorhersage
#   4. Kümmert sich um Loss, Optimizer und Metriken (AUC-ROC, F1)
#
# Relevante Lektionen:
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
# ARCHITEKTUR-ÜBERSICHT (Multimodal Fusion)
#
#   Bild (224x224x3)
#       │
#   EfficientNet-B0 Backbone (CNN, vortrainiert auf ImageNet)
#       │
#   Global Average Pooling
#       │
#   Image Feature Vector  (1280-dim)
#       │                               Metadaten (Alter, Geschlecht, Körperstelle)
#       │                                   │
#       │                               Metadata MLP (kleines Netz)
#       │                                   │
#       │                               Metadata Feature Vector (32-dim)
#       │                                   │
#       └──────────── Konkatenation ────────┘
#                           │
#                   [1280 + 32] = 1312-dim
#                           │
#                   Classifier Head (FC-Layer)
#                           │
#                   Output: 1 Wert (logit → Wahrscheinlichkeit Melanom)
#
# =============================================================================


class MelanomaModel(L.LightningModule):

    def __init__(self, metadata_dim=9, learning_rate=1e-4, pos_weight=50.0, dropout=0.3):
        """
        metadata_dim  : Anzahl Input-Features nach Preprocessing der Metadaten
                        Alter (1) + Geschlecht one-hot (2) + Körperstelle one-hot (6) = 9
        learning_rate : Lernrate für Adam Optimizer
        pos_weight    : Gewicht für positive Klasse (Melanom) im Loss
                        Faustregel: neg/pos = 98/2 ≈ 50
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
        # 2. METADATA MLP (kleines Netz nur für Metadaten)
        #
        # Die Metadaten (Alter, Geschlecht, Körperstelle) sind numerisch/kategorisch.
        # Ein kleines Fully-Connected-Netz verarbeitet diese separat zu einem
        # kompakten Feature-Vektor (32-dim), bevor wir ihn mit den Bild-Features fusionieren.
        #
        # Warum nicht direkt zusammenwerfen?
        # → Das Bild hat 1280 Werte, Metadaten nur 9. Direktes Concatenate würde
        #   die Metadaten "begraben". Das MLP gibt ihnen einen eigenen Lernpfad.
        # ------------------------------------------------------------------
        self.metadata_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 64),   # 9 → 64
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 32),             # 64 → 32
            nn.ReLU()
        )
        metadata_out = 32  # Ausgabegrösse des Metadata MLP

        # ------------------------------------------------------------------
        # 3. FUSION CLASSIFIER HEAD
        #
        # Hier werden Bild-Features (1280) und Metadaten-Features (32) zusammengeführt.
        # Konkatenation: einfach nebeneinanderlegen → Vektor der Länge 1280+32 = 1312
        #
        # Danach: zwei FC-Layer die den kombinierten Vektor auf 1 Wert reduzieren.
        # Dieser 1 Wert ist der "Logit" – via Sigmoid wird er zur Wahrscheinlichkeit.
        # (Kein Sigmoid hier, weil BCEWithLogitsLoss das intern macht → numerisch stabiler)
        # ------------------------------------------------------------------
        fusion_in = backbone_out + metadata_out  # 1280 + 32 = 1312

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(fusion_in, 256),     # 1312 → 256
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)              # 256 → 1 (Logit, kein Sigmoid!)
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

        # Metadaten durch MLP → Feature-Vektor [batch, 32]
        meta_features = self.metadata_mlp(metadata)

        # Beide Vektoren nebeneinander hängen → [batch, 1312]
        combined = torch.cat([image_features, meta_features], dim=1)

        # Durch Classifier → [batch, 1] (ein Logit pro Sample)
        logits = self.classifier(combined)
        return logits.squeeze(1)  # → [batch] (1D)

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
