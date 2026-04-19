# =============================================================================
# model.py
# Zuständig für: Modellarchitektur, Loss, Optimizer, Metriken
# Lesson 3: Transfer Learning, Regularization (Dropout)
# Lesson 4: CNNs
# Lesson 5: Residual Networks (EfficientNet baut darauf auf)
# =============================================================================

import torch
import torch.nn as nn
import lightning as L
import timm  # Library mit vielen vortrainierten Modellen (EfficientNet, ResNet, ViT, ...)

from torchmetrics import AUROC, F1Score, Accuracy
from torchmetrics.classification import BinaryConfusionMatrix


# =============================================================================
# MODELL KLASSE (Lightning)
# LightningModule kapselt: forward pass, loss, optimizer, metriken
# Lightning ruft training_step(), validation_step() etc. automatisch auf
# =============================================================================

class MelanomaModel(L.LightningModule):
    def __init__(self, learning_rate=1e-4, pos_weight=50.0, dropout=0.3):
        """
        learning_rate : Lernrate für den Optimizer (Lesson 3 - Hyperparameter Tuning)
        pos_weight    : Gewicht für positive Klasse im Loss (Class Imbalance)
                        Faustregel: ~98/2 = 49 → wir nehmen 50
        dropout       : Dropout-Rate (Lesson 3 - Regularization)
        """
        super().__init__()
        self.save_hyperparameters()  # speichert alle __init__ Parameter automatisch

        # -------------------------------------------------------
        # BACKBONE: EfficientNet-B0 (Transfer Learning)
        # (Lesson 3 - Transfer Learning, Lesson 4 - CNNs, Lesson 5 - ResNets)
        #
        # Was ist Transfer Learning?
        # EfficientNet wurde auf ImageNet (1.2 Mio Bilder, 1000 Klassen) vortrainiert.
        # Es hat bereits gelernt: Kanten, Texturen, Formen, komplexe Muster erkennen.
        # Wir nehmen diese "Basis" und passen sie an unsere Aufgabe (Melanom) an.
        # → Viel besser als von Null trainieren, besonders mit begrenzten Daten!
        #
        # EfficientNet Besonderheit:
        # Skaliert Width, Depth und Resolution gleichzeitig → sehr effizient
        # B0 = kleinste Variante, ideal zum Starten
        # -------------------------------------------------------
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,    # ImageNet Gewichte laden (Transfer Learning!)
            num_classes=0,      # Classifier-Head entfernen (wir fügen unseren eigenen hinzu)
            global_pool="avg"   # Global Average Pooling nach dem letzten Conv-Layer
        )

        # Grösse des Feature-Vektors aus dem Backbone (EfficientNet-B0 → 1280)
        backbone_out = self.backbone.num_features  # = 1280

        # -------------------------------------------------------
        # CLASSIFIER HEAD
        # Nimmt die 1280 Features vom Backbone und macht daraus eine Vorhersage
        # Dropout = Regularisierung gegen Overfitting (Lesson 3 - Dropout)
        # -------------------------------------------------------
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),          # Zufällig Neuronen deaktivieren → robuster
            nn.Linear(backbone_out, 256),   # Fully Connected Layer: 1280 → 256
            nn.ReLU(),                      # Aktivierungsfunktion (nicht-linear)
            nn.Dropout(p=dropout),
            nn.Linear(256, 1)               # Output: 1 Wert (Wahrscheinlichkeit Melanom)
            # Kein Sigmoid hier! BCEWithLogitsLoss macht das intern (numerisch stabiler)
        )

        # -------------------------------------------------------
        # LOSS FUNKTION: Binary Cross Entropy mit pos_weight
        # (Lesson 2 - Loss Functions)
        #
        # BCEWithLogitsLoss = BCE + Sigmoid kombiniert (stabiler als getrennt)
        # pos_weight: positive Samples (Melanom) werden stärker gewichtet
        # → Modell wird stärker bestraft wenn es ein Melanom verpasst
        # -------------------------------------------------------
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

        # -------------------------------------------------------
        # METRIKEN
        # Accuracy allein ist bei Class Imbalance irreführend!
        # (98% Accuracy möglich wenn man einfach immer "negativ" sagt)
        # AUC-ROC ist die wichtigste Metrik bei diesem Kaggle-Wettbewerb
        # -------------------------------------------------------
        self.train_auc = AUROC(task="binary")
        self.val_auc   = AUROC(task="binary")
        self.val_f1    = F1Score(task="binary", threshold=0.5)

    def forward(self, x):
        """
        Forward Pass: Bild rein → Vorhersage raus
        x: Batch von Bildern, Shape: [batch_size, 3, 224, 224]
        """
        features = self.backbone(x)      # CNN extrahiert Features: [batch_size, 1280]
        logits = self.classifier(features)  # Classifier: [batch_size, 1]
        return logits.squeeze(1)         # → [batch_size]

    def training_step(self, batch, batch_idx):
        """
        Wird von Lightning für jeden Trainings-Batch aufgerufen.
        Berechnet Loss und Metriken für einen Batch.
        """
        images, labels = batch
        logits = self(images)            # Forward Pass

        # Loss berechnen
        loss = self.criterion(logits, labels)

        # Wahrscheinlichkeiten für Metriken (Sigmoid wandelt Logits in 0-1 um)
        probs = torch.sigmoid(logits)
        self.train_auc.update(probs, labels.int())

        # Werte loggen (erscheinen in TensorBoard / W&B)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_train_epoch_end(self):
        # AUC am Ende jeder Epoche loggen und zurücksetzen
        self.log("train_auc", self.train_auc.compute(), prog_bar=True)
        self.train_auc.reset()

    def validation_step(self, batch, batch_idx):
        """
        Wird von Lightning für jeden Validation-Batch aufgerufen.
        Kein Gradient-Update hier! (torch.no_grad() macht Lightning automatisch)
        """
        images, labels = batch
        logits = self(images)

        loss = self.criterion(logits, labels)
        probs = torch.sigmoid(logits)

        self.val_auc.update(probs, labels.int())
        self.val_f1.update(probs, labels.int())

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def on_validation_epoch_end(self):
        # Metriken am Ende jeder Validation-Epoche loggen
        self.log("val_auc", self.val_auc.compute(), prog_bar=True)
        self.log("val_f1",  self.val_f1.compute(),  prog_bar=True)
        self.val_auc.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        """
        Wird von Lightning für den Test-Set aufgerufen (nur einmal am Ende).
        """
        images, labels = batch
        logits = self(images)
        probs = torch.sigmoid(logits)
        self.val_auc.update(probs, labels.int())
        self.val_f1.update(probs, labels.int())

    def on_test_epoch_end(self):
        self.log("test_auc", self.val_auc.compute())
        self.log("test_f1",  self.val_f1.compute())
        self.val_auc.reset()
        self.val_f1.reset()

    def configure_optimizers(self):
        """
        Definiert den Optimizer und optional einen Learning Rate Scheduler.
        (Lesson 3 - Hyperparameter Tuning)

        Adam = adaptiver Optimizer, sehr gut für Deep Learning
        ReduceLROnPlateau = halbiert LR wenn val_loss nicht mehr sinkt
        """
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-5   # L2 Regularisierung (Lesson 3 - Explicit Regularization)
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",        # val_loss soll minimiert werden
            factor=0.5,        # LR halbieren
            patience=3,        # nach 3 Epochen ohne Verbesserung
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss"  # beobachtet val_loss
            }
        }
