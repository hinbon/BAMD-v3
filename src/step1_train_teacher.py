#!/usr/bin/env python3

import argparse
import json
import os
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
)


# ============================================================
# 1. DATA SCHEMA
# ============================================================

CATEGORICAL_COLS = [
    "node_id",
    "parent_id",
    "rpl_ver",
]

LABEL_COL = "label"

UNKNOWN_CATEGORY = 0


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed: int = 42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. JSON ENCODER
# ============================================================

class NumpyJSONEncoder(json.JSONEncoder):

    def default(self, obj):

        if isinstance(obj, np.integer):
            return int(obj)

        if isinstance(obj, np.floating):
            return float(obj)

        if isinstance(obj, np.ndarray):
            return obj.tolist()

        return super().default(obj)


# ============================================================
# 4. PREPROCESSOR
# ============================================================

class TabularPreprocessor:

    """
    Preprocessing MUST be fitted using TRAIN only.

    Numerical:
        StandardScaler

    Categorical:
        integer mapping
        0 = unknown category

    Label:
        mapping to 0 ... C-1
    """

    def __init__(self):

        self.numeric_cols = None

        self.categorical_cols = (
            CATEGORICAL_COLS.copy()
        )

        self.scaler = StandardScaler()

        self.numeric_medians = {}

        self.category_maps = {}

        self.label_map = {}

        self.inverse_label_map = {}

    # --------------------------------------------------------
    # FIT
    # --------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame
    ):

        # ----------------------------------------------------
        # Determine numerical columns
        # ----------------------------------------------------

        self.numeric_cols = [
            col
            for col in train_df.columns
            if (
                col not in self.categorical_cols
                and col != LABEL_COL
            )
        ]

        # Check expected schema
        if len(self.numeric_cols) != 25:

            print(
                f"WARNING: detected "
                f"{len(self.numeric_cols)} numerical features."
            )

        # ----------------------------------------------------
        # Numerical preprocessing
        # ----------------------------------------------------

        numeric_df = train_df[
            self.numeric_cols
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        # Replace inf
        numeric_df = numeric_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        # Median imputation
        # Statistics come ONLY from TRAIN.
        for col in self.numeric_cols:

            median = numeric_df[col].median()

            if pd.isna(median):
                median = 0.0

            self.numeric_medians[col] = float(
                median
            )

            numeric_df[col] = (
                numeric_df[col]
                .fillna(median)
            )

        # Fit scaler ONLY on training data
        self.scaler.fit(
            numeric_df.values.astype(
                np.float32
            )
        )

        # ----------------------------------------------------
        # Categorical preprocessing
        # ----------------------------------------------------

        for col in self.categorical_cols:

            values = (
                train_df[col]
                .fillna("__MISSING__")
                .astype(str)
            )

            unique_values = sorted(
                values.unique()
            )

            # 0 is reserved for unknown
            mapping = {
                value: idx + 1
                for idx, value in enumerate(
                    unique_values
                )
            }

            self.category_maps[col] = mapping

        # ----------------------------------------------------
        # Label preprocessing
        # ----------------------------------------------------

        labels = sorted(
            train_df[LABEL_COL].unique()
        )

        self.label_map = {
            str(label): idx
            for idx, label in enumerate(labels)
        }

        self.inverse_label_map = {
            str(idx): str(label)
            for idx, label in enumerate(labels)
        }

        return self

    # --------------------------------------------------------
    # TRANSFORM
    # --------------------------------------------------------

    def transform(
        self,
        df: pd.DataFrame,
        with_label=True
    ):

        # ----------------------------------------------------
        # Numerical features
        # ----------------------------------------------------

        numeric_df = df[
            self.numeric_cols
        ].apply(
            pd.to_numeric,
            errors="coerce"
        )

        numeric_df = numeric_df.replace(
            [np.inf, -np.inf],
            np.nan
        )

        for col in self.numeric_cols:

            numeric_df[col] = (
                numeric_df[col]
                .fillna(
                    self.numeric_medians[col]
                )
            )

        X_num = self.scaler.transform(
            numeric_df.values.astype(
                np.float32
            )
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Categorical features
        # ----------------------------------------------------

        cat_arrays = []

        for col in self.categorical_cols:

            mapping = self.category_maps[col]

            values = (
                df[col]
                .fillna("__MISSING__")
                .astype(str)
            )

            encoded = (
                values
                .map(mapping)
                .fillna(UNKNOWN_CATEGORY)
                .astype(np.int64)
                .values
            )

            cat_arrays.append(
                encoded
            )

        X_cat = np.stack(
            cat_arrays,
            axis=1
        )

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        if not with_label:

            return X_num, X_cat

        y = np.array(
            [
                self.label_map[str(v)]
                for v in df[LABEL_COL]
            ],
            dtype=np.int64
        )

        return X_num, X_cat, y

    # --------------------------------------------------------
    # CARDINALITIES
    # --------------------------------------------------------

    def category_cardinalities(self):

        return [
            len(
                self.category_maps[col]
            ) + 1
            for col in self.categorical_cols
        ]

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    def save(
        self,
        output_dir: Path
    ):

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Numerical scaler
        joblib.dump(
            self.scaler,
            output_dir / "scaler.pkl"
        )

        metadata = {
            "numeric_cols":
                self.numeric_cols,

            "categorical_cols":
                self.categorical_cols,

            "numeric_medians":
                self.numeric_medians,

            "category_maps":
                self.category_maps,

            "label_map":
                self.label_map,

            "inverse_label_map":
                self.inverse_label_map,
        }

        with open(
            output_dir / "preprocessor.json",
            "w"
        ) as f:

            json.dump(
                metadata,
                f,
                indent=2,
                cls=NumpyJSONEncoder
            )


# ============================================================
# 5. DATASET
# ============================================================

class MixedTabularDataset(Dataset):

    def __init__(
        self,
        X_num,
        X_cat,
        y
    ):

        self.X_num = torch.from_numpy(
            X_num
        )

        self.X_cat = torch.from_numpy(
            X_cat
        )

        self.y = torch.from_numpy(
            y
        )

    def __len__(self):

        return len(self.y)

    def __getitem__(
        self,
        idx
    ):

        return (
            self.X_num[idx],
            self.X_cat[idx],
            self.y[idx]
        )


# ============================================================
# 6. TEACHER MODEL
# ============================================================

class TeacherMLP(nn.Module):

    def __init__(
        self,
        num_numeric,
        category_cardinalities,
        num_classes,
        embedding_dim=16,
        representation_dim=64,
        dropout=0.1
    ):

        super().__init__()

        # ----------------------------------------------------
        # Categorical embeddings
        # ----------------------------------------------------

        self.embeddings = nn.ModuleList()

        total_embedding_dim = 0

        for cardinality in category_cardinalities:

            emb_dim = min(
                embedding_dim,
                max(
                    4,
                    cardinality // 2
                )
            )

            self.embeddings.append(
                nn.Embedding(
                    num_embeddings=cardinality,
                    embedding_dim=emb_dim,
                    padding_idx=0
                )
            )

            total_embedding_dim += emb_dim

        # ----------------------------------------------------
        # Backbone
        # ----------------------------------------------------

        input_dim = (
            num_numeric
            + total_embedding_dim
        )

        self.backbone = nn.Sequential(

            nn.Linear(
                input_dim,
                256
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                128,
                representation_dim
            ),

            nn.ReLU(),
        )

        # ----------------------------------------------------
        # Classifier
        # ----------------------------------------------------

        self.classifier = nn.Linear(
            representation_dim,
            num_classes
        )

    # --------------------------------------------------------
    # ENCODE
    # --------------------------------------------------------

    def encode(
        self,
        X_num,
        X_cat
    ):

        embeddings = []

        for i, embedding in enumerate(
            self.embeddings
        ):

            emb = embedding(
                X_cat[:, i]
            )

            embeddings.append(
                emb
            )

        X = torch.cat(
            [
                X_num,
                *embeddings
            ],
            dim=1
        )

        h = self.backbone(X)

        return h

    # --------------------------------------------------------
    # FORWARD
    # --------------------------------------------------------

    def forward(
        self,
        X_num,
        X_cat
    ):

        h = self.encode(
            X_num,
            X_cat
        )

        logits = self.classifier(
            h
        )

        return logits, h


# ============================================================
# 7. EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    criterion
):

    model.eval()

    total_loss = 0.0
    total_n = 0

    all_preds = []
    all_targets = []

    for X_num, X_cat, y in loader:

        X_num = X_num.to(
            device,
            non_blocking=True
        )

        X_cat = X_cat.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        logits, _ = model(
            X_num,
            X_cat
        )

        loss = criterion(
            logits,
            y
        )

        n = y.size(0)

        total_loss += (
            loss.item() * n
        )

        total_n += n

        preds = torch.argmax(
            logits,
            dim=1
        )

        all_preds.append(
            preds.cpu().numpy()
        )

        all_targets.append(
            y.cpu().numpy()
        )

    all_preds = np.concatenate(
        all_preds
    )

    all_targets = np.concatenate(
        all_targets
    )

    avg_loss = (
        total_loss / total_n
    )

    accuracy = accuracy_score(
        all_targets,
        all_preds
    )

    macro_f1 = f1_score(
        all_targets,
        all_preds,
        average="macro"
    )

    return {
        "loss": float(avg_loss),
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "predictions": all_preds,
        "targets": all_targets
    }


# ============================================================
# 8. EXTRACT TEACHER ARTIFACTS
# ============================================================

@torch.no_grad()
def extract_teacher_artifacts(
    model,
    loader,
    device,
    output_dir
):

    model.eval()

    criterion = nn.CrossEntropyLoss(
        reduction="none"
    )

    all_hidden = []
    all_logits = []
    all_probs = []
    all_losses = []
    all_labels = []

    for X_num, X_cat, y in loader:

        X_num = X_num.to(
            device,
            non_blocking=True
        )

        X_cat = X_cat.to(
            device,
            non_blocking=True
        )

        y = y.to(
            device,
            non_blocking=True
        )

        logits, hidden = model(
            X_num,
            X_cat
        )

        probs = torch.softmax(
            logits,
            dim=1
        )

        sample_loss = criterion(
            logits,
            y
        )

        all_hidden.append(
            hidden.cpu().numpy().astype(
                np.float32
            )
        )

        all_logits.append(
            logits.cpu().numpy().astype(
                np.float32
            )
        )

        all_probs.append(
            probs.cpu().numpy().astype(
                np.float32
            )
        )

        all_losses.append(
            sample_loss.cpu().numpy().astype(
                np.float32
            )
        )

        all_labels.append(
            y.cpu().numpy().astype(
                np.int64
            )
        )

    hidden = np.concatenate(
        all_hidden,
        axis=0
    )

    logits = np.concatenate(
        all_logits,
        axis=0
    )

    probs = np.concatenate(
        all_probs,
        axis=0
    )

    losses = np.concatenate(
        all_losses,
        axis=0
    )

    labels = np.concatenate(
        all_labels,
        axis=0
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    np.save(
        output_dir / "train_hidden.npy",
        hidden
    )

    np.save(
        output_dir / "train_logits.npy",
        logits
    )

    np.save(
        output_dir / "train_probs.npy",
        probs
    )

    np.save(
        output_dir / "train_loss.npy",
        losses
    )

    np.save(
        output_dir / "train_labels.npy",
        labels
    )

    # Original row IDs
    np.save(
        output_dir / "train_indices.npy",
        np.arange(
            len(labels),
            dtype=np.int64
        )
    )

    print("\nTeacher artifacts:")
    print(
        f"  hidden : {hidden.shape}"
    )
    print(
        f"  logits : {logits.shape}"
    )
    print(
        f"  probs  : {probs.shape}"
    )
    print(
        f"  loss   : {losses.shape}"
    )
    print(
        f"  labels : {labels.shape}"
    )


# ============================================================
# 9. TRAIN TEACHER
# ============================================================

def train_teacher(
    model,
    train_loader,
    val_loader,
    device,
    output_dir,
    epochs,
    lr,
    weight_decay,
    early_stopping,
    amp
):

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2
        )
    )

    use_amp = (
        amp
        and device.type == "cuda"
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp
    )

    best_val_loss = float("inf")
    best_epoch = 0

    patience_counter = 0

    checkpoint_path = (
        output_dir / "teacher_best.pt"
    )

    for epoch in range(
        1,
        epochs + 1
    ):

        model.train()

        total_train_loss = 0.0
        total_train_n = 0

        for X_num, X_cat, y in train_loader:

            X_num = X_num.to(
                device,
                non_blocking=True
            )

            X_cat = X_cat.to(
                device,
                non_blocking=True
            )

            y = y.to(
                device,
                non_blocking=True
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            if use_amp:

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16
                ):

                    logits, _ = model(
                        X_num,
                        X_cat
                    )

                    loss = criterion(
                        logits,
                        y
                    )

            else:

                logits, _ = model(
                    X_num,
                    X_cat
                )

                loss = criterion(
                    logits,
                    y
                )

            scaler.scale(
                loss
            ).backward()

            scaler.step(
                optimizer
            )

            scaler.update()

            n = y.size(0)

            total_train_loss += (
                loss.item() * n
            )

            total_train_n += n

        train_loss = (
            total_train_loss
            / total_train_n
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate(
            model,
            val_loader,
            device,
            criterion
        )

        scheduler.step(
            val_metrics["loss"]
        )

        current_lr = (
            optimizer.param_groups[0]["lr"]
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss={train_loss:.6f} | "
            f"Val Loss={val_metrics['loss']:.6f} | "
            f"Val Acc={val_metrics['accuracy']:.6f} | "
            f"Val Macro-F1={val_metrics['macro_f1']:.6f} | "
            f"LR={current_lr:.2e}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if (
            val_metrics["loss"]
            < best_val_loss
        ):

            best_val_loss = (
                val_metrics["loss"]
            )

            best_epoch = epoch

            patience_counter = 0

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict":
                        model.state_dict(),
                    "best_val_loss":
                        best_val_loss,
                },
                checkpoint_path
            )

            print(
                "  -> saved best teacher"
            )

        else:

            patience_counter += 1

        if (
            patience_counter
            >= early_stopping
        ):

            print(
                f"\nEarly stopping at "
                f"epoch {epoch}"
            )

            break

    return (
        best_epoch,
        best_val_loss
    )


# ============================================================
# 10. MAIN
# ============================================================

def main(args):

    set_seed(args.seed)

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("TEACHER TRAINING")
    print("=" * 80)

    print(
        f"Device: {device}"
    )

    # ========================================================
    # [1/7] LOAD RAW DATA
    # ========================================================

    print(
        "\n[1/7] Loading raw data"
    )

    train_df = pd.read_csv(
        args.train
    )

    val_df = pd.read_csv(
        args.val
    )

    print(
        f"Train: {train_df.shape}"
    )

    print(
        f"Val  : {val_df.shape}"
    )

    # --------------------------------------------------------
    # Schema checks
    # --------------------------------------------------------

    if LABEL_COL not in train_df.columns:
        raise ValueError(
            f"Missing '{LABEL_COL}' in train data."
        )

    if LABEL_COL not in val_df.columns:
        raise ValueError(
            f"Missing '{LABEL_COL}' in validation data."
        )

    for col in CATEGORICAL_COLS:

        if col not in train_df.columns:
            raise ValueError(
                f"Missing categorical column "
                f"'{col}' in train data."
            )

    # --------------------------------------------------------
    # Check column equality
    # --------------------------------------------------------

    if list(train_df.columns) != list(
        val_df.columns
    ):

        raise ValueError(
            "Train and validation columns "
            "are not identical."
        )

    # ========================================================
    # Class distribution
    # ========================================================

    print(
        "\nClass distribution:"
    )

    train_counts = (
        train_df[LABEL_COL]
        .value_counts()
        .sort_index()
    )

    val_counts = (
        val_df[LABEL_COL]
        .value_counts()
        .sort_index()
    )

    print(
        "TRAIN:"
    )

    print(
        train_counts
    )

    print(
        "\nVAL:"
    )

    print(
        val_counts
    )

    # ========================================================
    # [2/7] FIT PREPROCESSING ON TRAIN ONLY
    # ========================================================

    print(
        "\n[2/7] Fitting preprocessing on TRAIN only"
    )

    preprocessor = (
        TabularPreprocessor()
    )

    preprocessor.fit(
        train_df
    )

    # --------------------------------------------------------
    # Transform train
    # --------------------------------------------------------

    (
        X_train_num,
        X_train_cat,
        y_train
    ) = preprocessor.transform(
        train_df,
        with_label=True
    )

    # --------------------------------------------------------
    # Transform validation
    # --------------------------------------------------------

    (
        X_val_num,
        X_val_cat,
        y_val
    ) = preprocessor.transform(
        val_df,
        with_label=True
    )

    # --------------------------------------------------------
    # Save preprocessing
    # --------------------------------------------------------

    preprocessor.save(
        output_dir
    )

    print(
        f"Numerical features: "
        f"{len(preprocessor.numeric_cols)}"
    )

    print(
        f"Categorical features: "
        f"{preprocessor.categorical_cols}"
    )

    print(
        f"Category cardinalities: "
        f"{preprocessor.category_cardinalities()}"
    )

    print(
        f"Number of classes: "
        f"{len(preprocessor.label_map)}"
    )

    # ========================================================
    # [3/7] DATASET / DATALOADER
    # ========================================================

    print(
        "\n[3/7] Creating DataLoaders"
    )

    train_dataset = (
        MixedTabularDataset(
            X_train_num,
            X_train_cat,
            y_train
        )
    )

    val_dataset = (
        MixedTabularDataset(
            X_val_num,
            X_val_cat,
            y_val
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        )
    )

    # Used ONLY to extract artifacts.
    # shuffle=False guarantees:
    #
    # hidden[i]
    # loss[i]
    # prob[i]
    # label[i]
    #
    # correspond to train.csv row i.

    train_extract_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        )
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
        persistent_workers=(
            args.num_workers > 0
        )
    )

    # ========================================================
    # [4/7] BUILD TEACHER
    # ========================================================

    print(
        "\n[4/7] Building teacher"
    )

    num_classes = len(
        np.unique(y_train)
    )

    model = TeacherMLP(
        num_numeric=len(
            preprocessor.numeric_cols
        ),
        category_cardinalities=(
            preprocessor.category_cardinalities()
        ),
        num_classes=num_classes,
        embedding_dim=16,
        representation_dim=64,
        dropout=args.dropout
    ).to(device)

    print(model)

    # ========================================================
    # [5/7] TRAIN
    # ========================================================

    print(
        "\n[5/7] Training teacher"
    )

    (
        best_epoch,
        best_val_loss
    ) = train_teacher(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        output_dir=output_dir,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        early_stopping=args.early_stopping,
        amp=args.amp
    )

    # ========================================================
    # LOAD BEST CHECKPOINT
    # ========================================================

    print(
        "\nLoading best checkpoint..."
    )

    checkpoint = torch.load(
        output_dir / "teacher_best.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ========================================================
    # [6/7] FINAL EVALUATION
    # ========================================================

    print(
        "\n[6/7] Final teacher evaluation"
    )

    criterion = nn.CrossEntropyLoss()

    train_metrics = evaluate(
        model,
        train_extract_loader,
        device,
        criterion
    )

    val_metrics = evaluate(
        model,
        val_loader,
        device,
        criterion
    )

    print(
        "\n================ TRAIN ================"
    )

    print(
        f"Loss     : "
        f"{train_metrics['loss']:.6f}"
    )

    print(
        f"Accuracy : "
        f"{train_metrics['accuracy']:.6f}"
    )

    print(
        f"Macro-F1 : "
        f"{train_metrics['macro_f1']:.6f}"
    )

    print(
        "\n================ VAL =================="
    )

    print(
        f"Loss     : "
        f"{val_metrics['loss']:.6f}"
    )

    print(
        f"Accuracy : "
        f"{val_metrics['accuracy']:.6f}"
    )

    print(
        f"Macro-F1 : "
        f"{val_metrics['macro_f1']:.6f}"
    )

    # --------------------------------------------------------
    # Detailed classification report
    # --------------------------------------------------------

    print(
        "\nValidation classification report:"
    )

    print(
        classification_report(
            val_metrics["targets"],
            val_metrics["predictions"],
            digits=6
        )
    )

    # ========================================================
    # [7/7] EXTRACT TRAIN ARTIFACTS
    # ========================================================

    print(
        "\n[7/7] Extracting teacher artifacts"
    )

    extract_teacher_artifacts(
        model=model,
        loader=train_extract_loader,
        device=device,
        output_dir=output_dir
    )

    # ========================================================
    # SAVE SUMMARY
    # ========================================================

    summary = {

        "train_path": args.train,

        "val_path": args.val,

        "train_samples":
            len(train_dataset),

        "val_samples":
            len(val_dataset),

        "numeric_columns":
            preprocessor.numeric_cols,

        "categorical_columns":
            preprocessor.categorical_cols,

        "num_numeric":
            len(preprocessor.numeric_cols),

        "num_categorical":
            len(preprocessor.categorical_cols),

        "category_cardinalities":
            preprocessor.category_cardinalities(),

        "num_classes":
            num_classes,

        "representation_dim":
            64,

        "best_epoch":
            best_epoch,

        "best_val_loss":
            best_val_loss,

        "train_metrics": {
            "loss":
                train_metrics["loss"],
            "accuracy":
                train_metrics["accuracy"],
            "macro_f1":
                train_metrics["macro_f1"]
        },

        "val_metrics": {
            "loss":
                val_metrics["loss"],
            "accuracy":
                val_metrics["accuracy"],
            "macro_f1":
                val_metrics["macro_f1"]
        },

        "seed":
            args.seed
    }

    with open(
        output_dir / "teacher_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
            cls=NumpyJSONEncoder
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "TEACHER TRAINING FINISHED"
    )

    print(
        "=" * 80

    )

    print(
        f"\nArtifacts saved to:"
        f"\n{output_dir}"
    )


# ============================================================
# 11. CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Train teacher MLP on raw "
            "mixed tabular data."
        )
    )

    parser.add_argument(
        "--train",
        required=True,
        help=(
            "RAW training CSV. "
            "Do not preprocess beforehand."
        )
    )

    parser.add_argument(
        "--val",
        required=True,
        help=(
            "RAW validation CSV. "
            "Do not preprocess beforehand."
        )
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=7
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=4
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use mixed precision on CUDA."
    )

    args = parser.parse_args()

    main(args)