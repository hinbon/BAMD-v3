#!/usr/bin/env python3

import argparse
import json
import random
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ============================================================
# CONFIG
# ============================================================

CATEGORICAL_COLS = [
    "node_id",
    "parent_id",
    "rpl_ver",
]

LABEL_COL = "label"


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# TEACHER MODEL
# Same architecture as train_teacher.py
# ============================================================

class TeacherMLP(nn.Module):

    def __init__(
        self,
        num_numeric,
        category_cardinalities,
        num_classes,
        embedding_dim=16,
        representation_dim=64,
        dropout=0.1,
    ):
        super().__init__()

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

        self.classifier = nn.Linear(
            representation_dim,
            num_classes
        )

    def encode(
        self,
        X_num,
        X_cat
    ):

        embeddings = []

        for i, embedding in enumerate(
            self.embeddings
        ):

            embeddings.append(
                embedding(
                    X_cat[:, i]
                )
            )

        X = torch.cat(
            [
                X_num,
                *embeddings
            ],
            dim=1
        )

        return self.backbone(X)

    def forward(
        self,
        X_num,
        X_cat
    ):

        h = self.encode(
            X_num,
            X_cat
        )

        logits = self.classifier(h)

        return logits, h


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

def load_preprocessor(
    output_dir
):

    output_dir = Path(output_dir)

    with open(
        output_dir / "preprocessor.json",
        "r"
    ) as f:

        metadata = json.load(f)

    scaler = joblib.load(
        output_dir / "scaler.pkl"
    )

    return metadata, scaler


# ============================================================
# TRANSFORM RAW DATA
# ============================================================

def transform_dataframe(
    df,
    metadata,
    scaler
):

    numeric_cols = (
        metadata["numeric_cols"]
    )

    categorical_cols = (
        metadata["categorical_cols"]
    )

    numeric_medians = (
        metadata["numeric_medians"]
    )

    category_maps = (
        metadata["category_maps"]
    )

    # --------------------------------------------------------
    # Numerical
    # --------------------------------------------------------

    X_num_df = df[
        numeric_cols
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_num_df = X_num_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    for col in numeric_cols:

        X_num_df[col] = (
            X_num_df[col]
            .fillna(
                numeric_medians[col]
            )
        )

    X_num = scaler.transform(
        X_num_df.values.astype(
            np.float32
        )
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Categorical
    # --------------------------------------------------------

    cat_arrays = []

    for col in categorical_cols:

        mapping = category_maps[col]

        values = (
            df[col]
            .fillna("__MISSING__")
            .astype(str)
        )

        encoded = (
            values
            .map(mapping)
            .fillna(0)
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

    # --------------------------------------------------------
    # Label
    # --------------------------------------------------------

    label_map = metadata["label_map"]

    y = np.array(
        [
            label_map[str(v)]
            for v in df[LABEL_COL]
        ],
        dtype=np.int64
    )

    return (
        X_num,
        X_cat,
        y
    )


# ============================================================
# MODEL INFERENCE
# ============================================================

@torch.no_grad()
def predict_loss(
    model,
    X_num,
    X_cat,
    y,
    device,
    batch_size=4096
):

    model.eval()

    dataset = TensorDataset(
        torch.from_numpy(X_num),
        torch.from_numpy(X_cat),
        torch.from_numpy(y)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda")
    )

    criterion = nn.CrossEntropyLoss(
        reduction="none"
    )

    all_losses = []

    for X_num_b, X_cat_b, y_b in loader:

        X_num_b = X_num_b.to(
            device,
            non_blocking=True
        )

        X_cat_b = X_cat_b.to(
            device,
            non_blocking=True
        )

        y_b = y_b.to(
            device,
            non_blocking=True
        )

        logits, _ = model(
            X_num_b,
            X_cat_b
        )

        losses = criterion(
            logits,
            y_b
        )

        all_losses.append(
            losses.cpu().numpy()
        )

    return np.concatenate(
        all_losses
    )


# ============================================================
# MAIN
# ============================================================

def main(args):

    set_seed(args.seed)

    output_dir = Path(
        args.teacher_dir
    )

    result_dir = Path(
        args.result_dir
    )

    result_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("=" * 80)
    print("PERMUTATION FEATURE IMPORTANCE")
    print("=" * 80)

    print(
        f"Device: {device}"
    )

    # ========================================================
    # 1. LOAD TRAIN DATA
    # ========================================================

    print(
        "\n[1/6] Loading RAW train data"
    )

    train_df = pd.read_csv(
        args.train
    )

    print(
        "Train:",
        train_df.shape
    )

    # ========================================================
    # 2. LOAD PREPROCESSOR
    # ========================================================

    print(
        "\n[2/6] Loading preprocessing"
    )

    metadata, scaler = (
        load_preprocessor(
            output_dir
        )
    )

    numeric_cols = (
        metadata["numeric_cols"]
    )

    categorical_cols = (
        metadata["categorical_cols"]
    )

    feature_cols = (
        numeric_cols
        + categorical_cols
    )

    print(
        f"Numerical features: "
        f"{len(numeric_cols)}"
    )

    print(
        f"Categorical features: "
        f"{len(categorical_cols)}"
    )

    print(
        f"Total features: "
        f"{len(feature_cols)}"
    )

    # ========================================================
    # 3. TRANSFORM
    # ========================================================

    print(
        "\n[3/6] Transforming train data"
    )

    X_num, X_cat, y = (
        transform_dataframe(
            train_df,
            metadata,
            scaler
        )
    )

    print(
        "X_num:",
        X_num.shape
    )

    print(
        "X_cat:",
        X_cat.shape
    )

    print(
        "y:",
        y.shape
    )

    # ========================================================
    # 4. LOAD TEACHER
    # ========================================================

    print(
        "\n[4/6] Loading teacher"
    )

    category_cardinalities = [
        len(
            metadata["category_maps"][col]
        ) + 1

        for col in categorical_cols
    ]

    num_classes = len(
        metadata["label_map"]
    )

    model = TeacherMLP(
        num_numeric=len(
            numeric_cols
        ),
        category_cardinalities=(
            category_cardinalities
        ),
        num_classes=num_classes,
        embedding_dim=16,
        representation_dim=64,
        dropout=args.dropout
    ).to(device)

    checkpoint = torch.load(
        output_dir / "teacher_best.pt",
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Teacher loaded."
    )

    print(
        f"Best epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    # ========================================================
    # 5. BASELINE
    # ========================================================

    print(
        "\n[5/6] Computing baseline loss"
    )

    baseline_losses = predict_loss(
        model=model,
        X_num=X_num,
        X_cat=X_cat,
        y=y,
        device=device,
        batch_size=args.batch_size
    )

    class_ids = np.unique(y)

    baseline_global = float(
        baseline_losses.mean()
    )

    baseline_class = {}

    for c in class_ids:

        mask = (
            y == c
        )

        baseline_class[int(c)] = float(
            baseline_losses[mask].mean()
        )

    print(
        f"Baseline global CE: "
        f"{baseline_global:.6f}"
    )

    for c in class_ids:

        mask = (
            y == c
        )

        print(
            f"Baseline class {c} CE: "
            f"{baseline_class[int(c)]:.6f} "
            f"(n={mask.sum()})"
        )

    # ========================================================
    # PERMUTATION
    # ========================================================

    print(
        "\n[6/6] Computing permutation importance"
    )

    rng = np.random.default_rng(
        args.seed
    )

    results = []

    # mapping:
    # categorical columns after numeric columns
    feature_locations = []

    for col in numeric_cols:

        feature_locations.append(
            (
                col,
                "numeric",
                numeric_cols.index(col)
            )
        )

    for col in categorical_cols:

        feature_locations.append(
            (
                col,
                "categorical",
                categorical_cols.index(col)
            )
        )

    for feature_idx, (
        feature_name,
        feature_type,
        local_idx
    ) in enumerate(
        feature_locations,
        start=1
    ):

        print(
            f"\n[{feature_idx}/{len(feature_locations)}] "
            f"{feature_name} ({feature_type})"
        )

        global_deltas = []
        class_deltas = {
            int(c): []
            for c in class_ids
        }

        for repeat in range(
            args.repeats
        ):

            # ------------------------------------------------
            # Copy only the relevant matrix
            # ------------------------------------------------

            if feature_type == "numeric":

                X_num_perm = X_num.copy()

                permutation = rng.permutation(
                    len(X_num_perm)
                )

                X_num_perm[:, local_idx] = (
                    X_num[
                        permutation,
                        local_idx
                    ]
                )

                X_cat_perm = X_cat

            else:

                X_cat_perm = X_cat.copy()

                permutation = rng.permutation(
                    len(X_cat_perm)
                )

                X_cat_perm[:, local_idx] = (
                    X_cat[
                        permutation,
                        local_idx
                    ]
                )

                X_num_perm = X_num

            # ------------------------------------------------
            # Predict
            # ------------------------------------------------

            perm_losses = predict_loss(
                model=model,
                X_num=X_num_perm,
                X_cat=X_cat_perm,
                y=y,
                device=device,
                batch_size=args.batch_size
            )

            # ------------------------------------------------
            # Global
            # ------------------------------------------------

            delta_global = float(
                perm_losses.mean()
                - baseline_global
            )

            global_deltas.append(
                delta_global
            )

            # ------------------------------------------------
            # Class-specific
            # ------------------------------------------------

            for c in class_ids:

                mask = (
                    y == c
                )

                base_c = (
                    baseline_class[int(c)]
                )

                perm_c = float(
                    perm_losses[mask].mean()
                )

                delta_c = (
                    perm_c - base_c
                )

                class_deltas[
                    int(c)
                ].append(
                    delta_c
                )

        # ----------------------------------------------------
        # Aggregate repeats
        # ----------------------------------------------------

        result = {
            "feature": feature_name,
            "type": feature_type,

            "global_mean":
                float(
                    np.mean(
                        global_deltas
                    )
                ),

            "global_std":
                float(
                    np.std(
                        global_deltas
                    )
                ),
        }

        for c in class_ids:

            values = np.array(
                class_deltas[int(c)]
            )

            result[
                f"class_{int(c)}_mean"
            ] = float(
                values.mean()
            )

            result[
                f"class_{int(c)}_std"
            ] = float(
                values.std()
            )

        results.append(
            result
        )

        print(
            f"  Global : "
            f"{result['global_mean']:.6f} "
            f"+/- "
            f"{result['global_std']:.6f}"
        )

        for c in class_ids:

            print(
                f"  Class {c}: "
                f"{result[f'class_{int(c)}_mean']:.6f} "
                f"+/- "
                f"{result[f'class_{int(c)}_std']:.6f}"
            )

    # ========================================================
    # SAVE
    # ========================================================

    result_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Ranking
    # --------------------------------------------------------

    result_df["global_rank"] = (
        result_df["global_mean"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    result_df["class_0_rank"] = (
        result_df["class_0_mean"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    result_df["class_1_rank"] = (
        result_df["class_1_mean"]
        .rank(
            ascending=False,
            method="min"
        )
        .astype(int)
    )

    # Save unsorted
    result_df.to_csv(
        result_dir /
        "feature_importance.csv",
        index=False
    )

    # Sorted versions
    result_df.sort_values(
        "global_mean",
        ascending=False
    ).to_csv(
        result_dir /
        "feature_importance_global_ranked.csv",
        index=False
    )

    result_df.sort_values(
        "class_0_mean",
        ascending=False
    ).to_csv(
        result_dir /
        "feature_importance_class0_ranked.csv",
        index=False
    )

    result_df.sort_values(
        "class_1_mean",
        ascending=False
    ).to_csv(
        result_dir /
        "feature_importance_class1_ranked.csv",
        index=False
    )

    # --------------------------------------------------------
    # Normalized importance
    # --------------------------------------------------------

    for col in [
        "global_mean",
        "class_0_mean",
        "class_1_mean"
    ]:

        # Negative permutation effects can happen
        # due to finite-sample / stochastic variation.
        positive = np.maximum(
            result_df[col].values,
            0.0
        )

        total = positive.sum()

        if total > 0:

            normalized = (
                positive / total
            )

        else:

            normalized = (
                np.ones_like(positive)
                / len(positive)
            )

        result_df[
            col.replace(
                "_mean",
                "_normalized"
            )
        ] = normalized

    result_df.to_csv(
        result_dir /
        "feature_importance.csv",
        index=False
    )

    # --------------------------------------------------------
    # Save baseline
    # --------------------------------------------------------

    baseline_summary = {

        "global_loss":
            baseline_global,

        "class_losses":
            baseline_class,

        "num_samples":
            int(len(y)),

        "num_features":
            int(len(feature_cols)),

        "repeats":
            int(args.repeats),

        "seed":
            int(args.seed),
    }

    with open(
        result_dir /
        "baseline.json",
        "w"
    ) as f:

        json.dump(
            baseline_summary,
            f,
            indent=2
        )

    print(
        "\n" + "=" * 80
    )

    print(
        "PERMUTATION IMPORTANCE FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        "\nSaved:"
    )

    print(
        result_dir /
        "feature_importance.csv"
    )

    print(
        "\nTop global features:"
    )

    print(
        result_df
        .sort_values(
            "global_mean",
            ascending=False
        )[
            [
                "feature",
                "global_mean",
                "class_0_mean",
                "class_1_mean"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )

    print(
        "\nTop class-1 features:"
    )

    print(
        result_df
        .sort_values(
            "class_1_mean",
            ascending=False
        )[
            [
                "feature",
                "global_mean",
                "class_0_mean",
                "class_1_mean"
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
        help="RAW train.csv"
    )

    parser.add_argument(
        "--teacher-dir",
        required=True,
        help="Directory containing teacher_best.pt"
    )

    parser.add_argument(
        "--result-dir",
        required=True
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096
    )

    parser.add_argument(
        "--repeats",
        type=int,
        default=5,
        help="Permutation repetitions per feature"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.1
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    main(args)