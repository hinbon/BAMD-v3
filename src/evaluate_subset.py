#!/usr/bin/env python3
"""
STEP 5B V3
===========

Fresh downstream MLP evaluation.

V3 keeps the V2 model/training protocol unchanged and adds support
for materialized subsets. A subset CSV can either:

    1) reference real rows from the original train.csv using
       source_train_index / source_train_position, OR
    2) contain the actual training rows directly.

The second mode is required for synthetic samples that do not exist
in the original train.csv.

Protocol
--------
Input:
    25 numerical features
    3 categorical features

Categorical embeddings:
    node_id   -> 16-D
    parent_id -> 16-D
    rpl_ver   -> 16-D

Total input:
    25 + 16 + 16 + 16 = 73

MLP:
    73 -> 256 -> 128 -> 64 -> 2

Training:
    Optimizer      : Adam
    LR             : 1e-3
    Weight decay   : 1e-4
    Batch size     : 256
    Epochs         : 100
    Early stopping : 20
    Loss           : CrossEntropyLoss
    Class weight   : None

Primary metric:
    Validation Macro-F1

Subset mapping:
    Mode A (real subset):
        source_train_index
        OR:
        source_train_position

    Mode B (materialized subset):
        the CSV directly contains the 25 numerical features,
        3 categorical features, and label/target.

    NEVER fallback to row order.

Scaler:
    Try joblib.load()
    Fallback to pickle.load()

Target:
    Supports:
        label
        target
"""

import os
import json
import pickle
import random
import argparse

from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# CONSTANTS
# ============================================================

CATEGORICAL_COLS = [
    "node_id",
    "parent_id",
    "rpl_ver",
]

NUMERICAL_COLS = [
    "time_sec",
    "rpl_rank",
    "dis_sent",
    "dio_sent",
    "dao_sent",
    "nbr_dis_rcv",
    "nbr_dio_rcv",
    "nbr_dao_ack_rcv",
    "nbr_fwd_to_me",
    "nbr_fwd_to_others",
    "nbr_fwd_bcast",
    "nbr_rpl_ctrl",
    "nbr_non_rpl_ctrl",
    "nbr_rpl_ver_rcv",
    "nbr_rpl_rank_rcv",
    "nbr_fwd_rpl",
    "nbr_fwd_non_rpl",
    "diff_rpl_rank",
    "diff_rpl_ver",
    "norm_rank_diff",
    "ctrl_to_data_ratio",
    "non_rpl_to_rpl_ratio",
    "rpl_fwd_ratio",
    "non_rpl_fwd_ratio",
    "total_fwd_ratio",
]

EMBED_DIM = 16

DEFAULT_BATCH_SIZE = 256
DEFAULT_EPOCHS = 100
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
DEFAULT_PATIENCE = 20

DEFAULT_SEEDS = [0, 1, 2]


# ============================================================
# SEED
# ============================================================

def set_seed(seed: int) -> None:

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# TARGET
# ============================================================

def resolve_target_column(
    df: pd.DataFrame,
    path: str,
) -> str:

    if "label" in df.columns:
        return "label"

    if "target" in df.columns:
        return "target"

    raise RuntimeError(
        f"\nTarget column not found in:\n"
        f"{path}\n\n"
        f"Expected one of:\n"
        f"    label\n"
        f"    target"
    )


def extract_target(
    df: pd.DataFrame,
    path: str,
) -> np.ndarray:

    col = resolve_target_column(
        df,
        path,
    )

    y = (
        pd.to_numeric(
            df[col],
            errors="raise",
        )
        .astype(np.int64)
        .to_numpy()
    )

    unique_values = np.unique(y)

    if not np.all(
        np.isin(
            unique_values,
            [0, 1],
        )
    ):

        raise RuntimeError(
            f"\nInvalid target values in {path}\n"
            f"Column: {col}\n"
            f"Values: {unique_values.tolist()}\n"
            f"Expected binary labels [0, 1]."
        )

    return y


# ============================================================
# SCALER
# ============================================================

def load_scaler(
    scaler_path: Path,
):
    """
    Load scaler.

    Priority:
        1. joblib
        2. pickle
    """

    if not scaler_path.exists():

        raise FileNotFoundError(
            f"\nScaler not found:\n"
            f"{scaler_path}"
        )

    print(
        f"\nLoading scaler:\n"
        f"{scaler_path}"
    )

    errors = []

    # --------------------------------------------------------
    # joblib
    # --------------------------------------------------------

    try:

        scaler = joblib.load(
            scaler_path
        )

        print(
            "Scaler loading method: joblib.load()"
        )

    except Exception as e:

        scaler = None

        errors.append(
            f"joblib.load -> {repr(e)}"
        )

    # --------------------------------------------------------
    # pickle
    # --------------------------------------------------------

    if scaler is None:

        try:

            with open(
                scaler_path,
                "rb",
            ) as f:

                scaler = pickle.load(f)

            print(
                "Scaler loading method: pickle.load()"
            )

        except Exception as e:

            errors.append(
                f"pickle.load -> {repr(e)}"
            )

    # --------------------------------------------------------
    # failed
    # --------------------------------------------------------

    if scaler is None:

        raise RuntimeError(
            "\nCould not load scaler.pkl\n\n"
            + "\n".join(errors)
        )

    print(
        f"Scaler type: "
        f"{type(scaler).__name__}"
    )

    if not hasattr(
        scaler,
        "transform",
    ):

        raise RuntimeError(
            "\nLoaded scaler does not have "
            ".transform().\n"
            f"Object type: {type(scaler)}"
        )

    return scaler


def verify_scaler(
    scaler,
    numerical_cols: List[str],
) -> None:

    print("\n")
    print("=" * 80)
    print("SCALER VERIFICATION")
    print("=" * 80)

    # --------------------------------------------------------
    # Number of features
    # --------------------------------------------------------

    if hasattr(
        scaler,
        "n_features_in_",
    ):

        n_features = int(
            scaler.n_features_in_
        )

        print(
            f"n_features_in_: "
            f"{n_features}"
        )

        if n_features != len(
            numerical_cols
        ):

            raise RuntimeError(
                f"\nScaler expects "
                f"{n_features} features.\n"
                f"Step 5B expects "
                f"{len(numerical_cols)}."
            )

    # --------------------------------------------------------
    # Feature names
    # --------------------------------------------------------

    if hasattr(
        scaler,
        "feature_names_in_",
    ):

        scaler_features = [
            str(x)
            for x in scaler.feature_names_in_
        ]

        expected_features = [
            str(x)
            for x in numerical_cols
        ]

        if (
            scaler_features
            != expected_features
        ):

            print(
                "\nScaler features:"
            )

            for i, name in enumerate(
                scaler_features,
                start=1,
            ):

                print(
                    f"  {i:02d}. {name}"
                )

            print(
                "\nExpected features:"
            )

            for i, name in enumerate(
                expected_features,
                start=1,
            ):

                print(
                    f"  {i:02d}. {name}"
                )

            raise RuntimeError(
                "\nScaler feature order mismatch."
            )

        print(
            "Feature order: PASS"
        )

    else:

        print(
            "Scaler has no feature_names_in_."
        )

        print(
            "Using dimensionality verification only."
        )

    print(
        "Scaler verification: PASS"
    )

    print("=" * 80)


# ============================================================
# CATEGORY ENCODER
# ============================================================

class CategoryEncoder:

    """
    Fit category mapping on the FULL original training set.

    0 = unknown
    1..K = known category
    """

    def __init__(self):

        self.vocabularies: Dict[
            str,
            Dict[str, int],
        ] = {}

        self.cardinalities: Dict[
            str,
            int,
        ] = {}

    @staticmethod
    def normalize(
        value,
    ) -> str:

        if pd.isna(value):
            return "__NA__"

        return str(value)

    def fit(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
    ):

        for col in categorical_cols:

            if col not in df.columns:

                raise RuntimeError(
                    f"Missing categorical "
                    f"column: {col}"
                )

            values = [
                self.normalize(v)
                for v in df[col]
            ]

            unique_values = sorted(
                set(values)
            )

            mapping = {
                value: idx + 1
                for idx, value
                in enumerate(
                    unique_values
                )
            }

            self.vocabularies[
                col
            ] = mapping

            # +1 for unknown index 0
            self.cardinalities[
                col
            ] = len(unique_values) + 1

        return self

    def transform(
        self,
        df: pd.DataFrame,
        categorical_cols: List[str],
    ) -> np.ndarray:

        columns = []

        for col in categorical_cols:

            mapping = (
                self.vocabularies[
                    col
                ]
            )

            encoded = [
                mapping.get(
                    self.normalize(v),
                    0,
                )
                for v in df[col]
            ]

            columns.append(
                np.asarray(
                    encoded,
                    dtype=np.int64,
                )
            )

        return np.stack(
            columns,
            axis=1,
        )


# ============================================================
# DATASET
# ============================================================

class MixedTabularDataset(
    Dataset
):

    def __init__(
        self,
        X_num: np.ndarray,
        X_cat: np.ndarray,
        y: np.ndarray,
    ):

        if not (
            len(X_num)
            == len(X_cat)
            == len(y)
        ):

            raise RuntimeError(
                "Dataset length mismatch."
            )

        self.X_num = torch.from_numpy(
            np.asarray(
                X_num,
                dtype=np.float32,
            )
        )

        self.X_cat = torch.from_numpy(
            np.asarray(
                X_cat,
                dtype=np.int64,
            )
        )

        self.y = torch.from_numpy(
            np.asarray(
                y,
                dtype=np.int64,
            )
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(
        self,
        index,
    ):

        return (
            self.X_num[index],
            self.X_cat[index],
            self.y[index],
        )


# ============================================================
# MODEL
# ============================================================

class MixedInputMLP(
    nn.Module
):

    def __init__(
        self,
        num_numerical_features: int,
        categorical_cardinalities: List[int],
        embedding_dim: int = EMBED_DIM,
    ):

        super().__init__()

        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(
                    num_embeddings=int(
                        cardinality
                    ),
                    embedding_dim=embedding_dim,
                )
                for cardinality
                in categorical_cardinalities
            ]
        )

        input_dim = (
            num_numerical_features
            + len(
                categorical_cardinalities
            )
            * embedding_dim
        )

        if input_dim != 73:

            raise RuntimeError(
                f"\nUnexpected model input dimension.\n"
                f"Numerical: "
                f"{num_numerical_features}\n"
                f"Categoricals: "
                f"{len(categorical_cardinalities)}\n"
                f"Embedding dim: "
                f"{embedding_dim}\n"
                f"Total: {input_dim}\n"
                f"Expected: 73"
            )

        self.classifier = nn.Sequential(

            nn.Linear(
                input_dim,
                256,
            ),

            nn.ReLU(),

            nn.Linear(
                256,
                128,
            ),

            nn.ReLU(),

            nn.Linear(
                128,
                64,
            ),

            nn.ReLU(),

            nn.Linear(
                64,
                2,
            ),
        )

    def forward(
        self,
        x_num,
        x_cat,
    ):

        embedded = []

        for i, embedding in enumerate(
            self.embeddings
        ):

            embedded.append(
                embedding(
                    x_cat[:, i]
                )
            )

        x = torch.cat(
            [x_num] + embedded,
            dim=1,
        )

        return self.classifier(x)


# ============================================================
# METRICS
# ============================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:

    return {
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "class1_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="binary",
                pos_label=1,
                zero_division=0,
            )
        ),
    }


# ============================================================
# EVALUATE
# ============================================================

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_true = []
    all_pred = []

    for (
        x_num,
        x_cat,
        y,
    ) in loader:

        x_num = x_num.to(
            device,
            non_blocking=True,
        )

        x_cat = x_cat.to(
            device,
            non_blocking=True,
        )

        y = y.to(
            device,
            non_blocking=True,
        )

        logits = model(
            x_num,
            x_cat,
        )

        loss = criterion(
            logits,
            y,
        )

        batch_size = y.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

        pred = torch.argmax(
            logits,
            dim=1,
        )

        all_true.append(
            y.cpu().numpy()
        )

        all_pred.append(
            pred.cpu().numpy()
        )

    y_true = np.concatenate(
        all_true
    )

    y_pred = np.concatenate(
        all_pred
    )

    metrics = compute_metrics(
        y_true,
        y_pred,
    )

    metrics["loss"] = (
        total_loss
        / total_samples
    )

    return metrics


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:

    model.eval()

    all_true = []
    all_pred = []

    for (
        x_num,
        x_cat,
        y,
    ) in loader:

        x_num = x_num.to(
            device,
            non_blocking=True,
        )

        x_cat = x_cat.to(
            device,
            non_blocking=True,
        )

        logits = model(
            x_num,
            x_cat,
        )

        pred = torch.argmax(
            logits,
            dim=1,
        )

        all_true.append(
            y.numpy()
        )

        all_pred.append(
            pred.cpu().numpy()
        )

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
    )


# ============================================================
# SUBSET LOADER
# ============================================================

def load_subset_indices(
    subset_path: str,
    train_df: pd.DataFrame,
) -> np.ndarray:

    subset_df = pd.read_csv(
        subset_path
    )

    # --------------------------------------------------------
    # Resolve source index
    # --------------------------------------------------------

    if (
        "source_train_index"
        in subset_df.columns
    ):

        indices = (
            pd.to_numeric(
                subset_df[
                    "source_train_index"
                ],
                errors="raise",
            )
            .astype(np.int64)
            .to_numpy()
        )

        index_column = (
            "source_train_index"
        )

    elif (
        "source_train_position"
        in subset_df.columns
    ):

        indices = (
            pd.to_numeric(
                subset_df[
                    "source_train_position"
                ],
                errors="raise",
            )
            .astype(np.int64)
            .to_numpy()
        )

        index_column = (
            "source_train_position"
        )

    else:

        raise RuntimeError(
            f"\nSubset file has no valid "
            f"source index column:\n"
            f"{subset_path}\n\n"
            f"Required:\n"
            f"  source_train_index\n"
            f"  OR\n"
            f"  source_train_position\n\n"
            f"Refusing to use row order."
        )

    # --------------------------------------------------------
    # Empty
    # --------------------------------------------------------

    if len(indices) == 0:

        raise RuntimeError(
            f"\nSubset is empty:\n"
            f"{subset_path}"
        )

    # --------------------------------------------------------
    # Duplicate
    # --------------------------------------------------------

    if (
        len(indices)
        != len(np.unique(indices))
    ):

        raise RuntimeError(
            f"\nDuplicate source indices:\n"
            f"{subset_path}\n"
            f"Rows   = {len(indices)}\n"
            f"Unique = {len(np.unique(indices))}"
        )

    # --------------------------------------------------------
    # Range
    # --------------------------------------------------------

    if np.any(indices < 0):

        raise RuntimeError(
            f"\nNegative source indices found:\n"
            f"{subset_path}"
        )

    if np.any(
        indices >= len(train_df)
    ):

        raise RuntimeError(
            f"\nOut-of-range source indices:\n"
            f"{subset_path}\n"
            f"Train size = {len(train_df)}\n"
            f"Min index  = {indices.min()}\n"
            f"Max index  = {indices.max()}"
        )

    # --------------------------------------------------------
    # Verify target
    # --------------------------------------------------------

    if "label" in subset_df.columns:

        subset_target_col = "label"

    elif "target" in subset_df.columns:

        subset_target_col = "target"

    else:

        raise RuntimeError(
            f"\nSubset contains neither "
            f"'label' nor 'target':\n"
            f"{subset_path}"
        )

    subset_y = (
        pd.to_numeric(
            subset_df[
                subset_target_col
            ],
            errors="raise",
        )
        .astype(np.int64)
        .to_numpy()
    )

    train_target_col = (
        resolve_target_column(
            train_df,
            "train.csv",
        )
    )

    train_y = (
        pd.to_numeric(
            train_df.iloc[
                indices
            ][
                train_target_col
            ],
            errors="raise",
        )
        .astype(np.int64)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Check target length
    # --------------------------------------------------------

    if len(subset_y) != len(
        train_y
    ):

        raise RuntimeError(
            "\nTarget length mismatch.\n"
            f"Subset = {len(subset_y)}\n"
            f"Train  = {len(train_y)}"
        )

    # --------------------------------------------------------
    # Check target values
    # --------------------------------------------------------

    if not np.array_equal(
        subset_y,
        train_y,
    ):

        mismatch = np.where(
            subset_y != train_y
        )[0]

        print(
            "\nTarget mismatch detected!"
        )

        for pos in mismatch[:10]:

            print(
                f"position={pos}, "
                f"source={indices[pos]}, "
                f"subset={subset_y[pos]}, "
                f"train={train_y[pos]}"
            )

        raise RuntimeError(
            f"\nTarget verification FAILED:\n"
            f"{subset_path}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    values, counts = np.unique(
        train_y,
        return_counts=True,
    )

    distribution = {
        int(v): int(c)
        for v, c
        in zip(values, counts)
    }

    print("\n")
    print("=" * 80)
    print("SUBSET VALIDATION")
    print("=" * 80)

    print(
        f"File           : "
        f"{subset_path}"
    )

    print(
        f"Index column   : "
        f"{index_column}"
    )

    print(
        f"N samples      : "
        f"{len(indices)}"
    )

    print(
        f"Unique indices : "
        f"{len(np.unique(indices))}"
    )

    print(
        f"Index range    : "
        f"[{indices.min()}, "
        f"{indices.max()}]"
    )

    print(
        f"Class dist     : "
        f"{distribution}"
    )

    print(
        "Target verification: PASS"
    )

    print("=" * 80)

    return indices



# ============================================================
# SUBSET SPECIFICATION
# ============================================================

def load_subset_spec(
    subset_path: str,
    train_df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
) -> Dict:
    """
    Load a subset in one of two modes.

    Mode A:
        source_train_index / source_train_position
        -> real rows from original train.csv.

    Mode B:
        materialized subset
        -> actual rows are already present in the CSV.
           This supports synthetic samples that do not exist
           in original train.csv.
    """

    subset_df = pd.read_csv(
        subset_path
    )

    # --------------------------------------------------------
    # Existing V2 indexed-subset path
    # --------------------------------------------------------

    if (
        "source_train_index" in subset_df.columns
        or "source_train_position" in subset_df.columns
    ):
        indices = load_subset_indices(
            subset_path,
            train_df,
        )

        return {
            "mode": "index",
            "path": str(subset_path),
            "indices": indices,
            "df": None,
        }

    # --------------------------------------------------------
    # Materialized-subset path
    # --------------------------------------------------------

    required_columns = (
        list(numerical_cols)
        + list(categorical_cols)
    )

    missing = [
        c
        for c in required_columns
        if c not in subset_df.columns
    ]

    if missing:
        raise RuntimeError(
            f"\nSubset is neither an indexed subset nor a "
            f"materialized subset.\n"
            f"File: {subset_path}\n\n"
            f"Missing feature columns:\n"
            f"{missing}\n\n"
            f"Expected numerical columns:\n"
            f"{numerical_cols}\n\n"
            f"Expected categorical columns:\n"
            f"{categorical_cols}"
        )

    target_col = resolve_target_column(
        subset_df,
        subset_path,
    )

    subset_y = (
        pd.to_numeric(
            subset_df[target_col],
            errors="raise",
        )
        .astype(np.int64)
        .to_numpy()
    )

    unique_values = np.unique(
        subset_y
    )

    if not np.all(
        np.isin(
            unique_values,
            [0, 1],
        )
    ):
        raise RuntimeError(
            f"\nInvalid target values in materialized subset:\n"
            f"{subset_path}\n"
            f"Values: {unique_values.tolist()}\n"
            f"Expected binary labels [0, 1]."
        )

    if len(subset_df) == 0:
        raise RuntimeError(
            f"\nMaterialized subset is empty:\n"
            f"{subset_path}"
        )

    values, counts = np.unique(
        subset_y,
        return_counts=True,
    )

    distribution = {
        int(v): int(c)
        for v, c in zip(values, counts)
    }

    print("\n")
    print("=" * 80)
    print("MATERIALIZED SUBSET VALIDATION")
    print("=" * 80)

    print(
        f"File           : {subset_path}"
    )

    print(
        "Subset mode    : materialized"
    )

    print(
        f"N samples      : {len(subset_df)}"
    )

    print(
        f"Class dist     : {distribution}"
    )

    print(
        "Target verification: PASS"
    )

    print("=" * 80)

    return {
        "mode": "materialized",
        "path": str(subset_path),
        "indices": None,
        "df": subset_df,
        "target_col": target_col,
    }


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_indices: np.ndarray,
    scaler,
    categorical_cols: List[str],
    numerical_cols: List[str],
):

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    for name, df in [
        ("train", train_df),
        ("val", val_df),
        ("test", test_df),
    ]:

        for col in numerical_cols:

            if col not in df.columns:

                raise RuntimeError(
                    f"\nMissing numerical feature "
                    f"'{col}' in {name}."
                )

        for col in categorical_cols:

            if col not in df.columns:

                raise RuntimeError(
                    f"\nMissing categorical feature "
                    f"'{col}' in {name}."
                )

    # --------------------------------------------------------
    # Category encoder
    # --------------------------------------------------------

    encoder = CategoryEncoder()

    encoder.fit(
        train_df,
        categorical_cols,
    )

    # --------------------------------------------------------
    # Numerical features
    # IMPORTANT:
    # Convert to numpy BEFORE scaler.transform()
    # --------------------------------------------------------

    train_numeric_raw = (
        train_df[
            numerical_cols
        ]
        .to_numpy()
    )

    val_numeric_raw = (
        val_df[
            numerical_cols
        ]
        .to_numpy()
    )

    test_numeric_raw = (
        test_df[
            numerical_cols
        ]
        .to_numpy()
    )

    try:

        X_train_num_full = (
            scaler.transform(
                train_numeric_raw
            )
        )

        X_val_num = (
            scaler.transform(
                val_numeric_raw
            )
        )

        X_test_num = (
            scaler.transform(
                test_numeric_raw
            )
        )

    except Exception as e:

        raise RuntimeError(
            "\nScaler transform failed.\n\n"
            f"Numerical feature count: "
            f"{len(numerical_cols)}\n"
            f"Scaler error:\n{e}"
        )

    # --------------------------------------------------------
    # Force float32
    # --------------------------------------------------------

    X_train_num_full = np.asarray(
        X_train_num_full,
        dtype=np.float32,
    )

    X_val_num = np.asarray(
        X_val_num,
        dtype=np.float32,
    )

    X_test_num = np.asarray(
        X_test_num,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    X_train_cat_full = (
        encoder.transform(
            train_df,
            categorical_cols,
        )
    )

    X_val_cat = (
        encoder.transform(
            val_df,
            categorical_cols,
        )
    )

    X_test_cat = (
        encoder.transform(
            test_df,
            categorical_cols,
        )
    )

    # --------------------------------------------------------
    # Selected train subset
    # --------------------------------------------------------

    X_train_num = (
        X_train_num_full[
            train_indices
        ]
    )

    X_train_cat = (
        X_train_cat_full[
            train_indices
        ]
    )

    y_full_train = (
        extract_target(
            train_df,
            "train.csv",
        )
    )

    y_train = (
        y_full_train[
            train_indices
        ]
    )

    y_val = (
        extract_target(
            val_df,
            "val.csv",
        )
    )

    y_test = (
        extract_target(
            test_df,
            "test.csv",
        )
    )

    return (
        X_train_num,
        X_train_cat,
        y_train,

        X_val_num,
        X_val_cat,
        y_val,

        X_test_num,
        X_test_cat,
        y_test,

        encoder.cardinalities,
    )



# ============================================================
# PREPARE MATERIALIZED SUBSET FEATURES
# ============================================================

def prepare_features_materialized(
    subset_df: pd.DataFrame,
    subset_path: str,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler,
    categorical_cols: List[str],
    numerical_cols: List[str],
):
    """
    Prepare a materialized subset directly.

    Important:
        - scaler comes from the fixed teacher/full-train artifact.
        - categorical vocabulary is fitted on original train_df,
          exactly like V2.
        - only the selected/materialized rows are used as the MLP
          training set.
    """

    # --------------------------------------------------------
    # Check columns
    # --------------------------------------------------------

    for col in numerical_cols:
        if col not in subset_df.columns:
            raise RuntimeError(
                f"\nMissing numerical feature "
                f"'{col}' in materialized subset.\n"
                f"File: {subset_path}"
            )

    for col in categorical_cols:
        if col not in subset_df.columns:
            raise RuntimeError(
                f"\nMissing categorical feature "
                f"'{col}' in materialized subset.\n"
                f"File: {subset_path}"
            )

    # --------------------------------------------------------
    # Category encoder
    #
    # FIT ON ORIGINAL TRAIN ONLY.
    # --------------------------------------------------------

    encoder = CategoryEncoder()

    encoder.fit(
        train_df,
        categorical_cols,
    )

    # --------------------------------------------------------
    # Numerical features
    # --------------------------------------------------------

    subset_numeric_raw = (
        subset_df[
            numerical_cols
        ]
        .to_numpy()
    )

    val_numeric_raw = (
        val_df[
            numerical_cols
        ]
        .to_numpy()
    )

    test_numeric_raw = (
        test_df[
            numerical_cols
        ]
        .to_numpy()
    )

    try:

        X_train_num = scaler.transform(
            subset_numeric_raw
        )

        X_val_num = scaler.transform(
            val_numeric_raw
        )

        X_test_num = scaler.transform(
            test_numeric_raw
        )

    except Exception as e:

        raise RuntimeError(
            "\nScaler transform failed for "
            "materialized subset.\n\n"
            f"Subset: {subset_path}\n"
            f"Numerical feature count: "
            f"{len(numerical_cols)}\n"
            f"Scaler error:\n{e}"
        )

    X_train_num = np.asarray(
        X_train_num,
        dtype=np.float32,
    )

    X_val_num = np.asarray(
        X_val_num,
        dtype=np.float32,
    )

    X_test_num = np.asarray(
        X_test_num,
        dtype=np.float32,
    )

    # --------------------------------------------------------
    # Categorical features
    # --------------------------------------------------------

    X_train_cat = encoder.transform(
        subset_df,
        categorical_cols,
    )

    X_val_cat = encoder.transform(
        val_df,
        categorical_cols,
    )

    X_test_cat = encoder.transform(
        test_df,
        categorical_cols,
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    y_train = extract_target(
        subset_df,
        subset_path,
    )

    y_val = extract_target(
        val_df,
        "val.csv",
    )

    y_test = extract_target(
        test_df,
        "test.csv",
    )

    return (
        X_train_num,
        X_train_cat,
        y_train,

        X_val_num,
        X_val_cat,
        y_val,

        X_test_num,
        X_test_cat,
        y_test,

        encoder.cardinalities,
    )


# ============================================================
# TRAIN ONE SEED
# ============================================================

def train_one_seed(
    X_train_num,
    X_train_cat,
    y_train,

    X_val_num,
    X_val_cat,
    y_val,

    X_test_num,
    X_test_cat,
    y_test,

    categorical_cardinality_list,

    seed,
    device,

    batch_size,
    max_epochs,
    lr,
    weight_decay,
    patience,

    checkpoint_path,
):

    print("\n")
    print("-" * 100)
    print(
        f"TRAINING SEED {seed}"
    )
    print("-" * 100)

    set_seed(seed)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = (
        MixedTabularDataset(
            X_train_num,
            X_train_cat,
            y_train,
        )
    )

    val_dataset = (
        MixedTabularDataset(
            X_val_num,
            X_val_cat,
            y_val,
        )
    )

    test_dataset = (
        MixedTabularDataset(
            X_test_num,
            X_test_cat,
            y_test,
        )
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    generator = torch.Generator()

    generator.manual_seed(
        seed
    )

    use_pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_pin_memory,
        generator=generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_pin_memory,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = MixedInputMLP(
        num_numerical_features=25,
        categorical_cardinalities=(
            categorical_cardinality_list
        ),
        embedding_dim=EMBED_DIM,
    ).to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = (
        nn.CrossEntropyLoss()
    )

    # --------------------------------------------------------
    # Early stopping
    # --------------------------------------------------------

    best_val_macro_f1 = (
        -float("inf")
    )

    best_epoch = 0

    epochs_without_improvement = 0

    best_state = None

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(
        1,
        max_epochs + 1,
    ):

        model.train()

        running_loss = 0.0
        num_train_samples = 0

        train_true = []
        train_pred = []

        # ----------------------------------------------------
        # Training batches
        # ----------------------------------------------------

        for (
            x_num,
            x_cat,
            y,
        ) in train_loader:

            x_num = x_num.to(
                device,
                non_blocking=True,
            )

            x_cat = x_cat.to(
                device,
                non_blocking=True,
            )

            y = y.to(
                device,
                non_blocking=True,
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                x_num,
                x_cat,
            )

            loss = criterion(
                logits,
                y,
            )

            loss.backward()

            optimizer.step()

            n = y.size(0)

            running_loss += (
                loss.item() * n
            )

            num_train_samples += n

            pred = torch.argmax(
                logits,
                dim=1,
            )

            train_true.append(
                y.detach()
                .cpu()
                .numpy()
            )

            train_pred.append(
                pred.detach()
                .cpu()
                .numpy()
            )

        train_loss = (
            running_loss
            / num_train_samples
        )

        train_true_np = np.concatenate(
            train_true
        )

        train_pred_np = np.concatenate(
            train_pred
        )

        train_metrics = (
            compute_metrics(
                train_true_np,
                train_pred_np,
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        current_score = (
            val_metrics[
                "macro_f1"
            ]
        )

        if (
            current_score
            > best_val_macro_f1
            + 1e-8
        ):

            best_val_macro_f1 = (
                current_score
            )

            best_epoch = epoch

            epochs_without_improvement = 0

            best_state = {
                key: value.detach()
                .cpu()
                .clone()
                for key, value
                in model.state_dict().items()
            }

            if checkpoint_path:

                torch.save(
                    {
                        "epoch":
                            int(epoch),

                        "val_macro_f1":
                            float(
                                best_val_macro_f1
                            ),

                        "model_state_dict":
                            best_state,
                    },
                    checkpoint_path,
                )

        else:

            epochs_without_improvement += 1

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss "
            f"{train_loss:.5f} | "
            f"Train Acc "
            f"{train_metrics['accuracy']:.5f} | "
            f"Train Macro-F1 "
            f"{train_metrics['macro_f1']:.5f} | "
            f"Train C1-F1 "
            f"{train_metrics['class1_f1']:.5f} | "
            f"Val Loss "
            f"{val_metrics['loss']:.5f} | "
            f"Val Acc "
            f"{val_metrics['accuracy']:.5f} | "
            f"Val Macro-F1 "
            f"{val_metrics['macro_f1']:.5f} | "
            f"Val C1-F1 "
            f"{val_metrics['class1_f1']:.5f}"
        )

        # ----------------------------------------------------
        # Stop
        # ----------------------------------------------------

        if (
            epochs_without_improvement
            >= patience
        ):

            print(
                f"\nEarly stopping at epoch "
                f"{epoch}."
            )

            print(
                f"Best epoch = "
                f"{best_epoch}"
            )

            print(
                f"Best Val Macro-F1 = "
                f"{best_val_macro_f1:.6f}"
            )

            break

    # ========================================================
    # Ensure best state
    # ========================================================

    if best_state is None:

        raise RuntimeError(
            "Training produced no best model."
        )

    model.load_state_dict(
        best_state
    )

    # ========================================================
    # Final validation/test
    # ========================================================

    final_val = evaluate(
        model,
        val_loader,
        criterion,
        device,
    )

    final_test = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    # ========================================================
    # Confusion matrices
    # ========================================================

    val_true, val_pred = (
        collect_predictions(
            model,
            val_loader,
            device,
        )
    )

    test_true, test_pred = (
        collect_predictions(
            model,
            test_loader,
            device,
        )
    )

    val_cm = confusion_matrix(
        val_true,
        val_pred,
        labels=[0, 1],
    )

    test_cm = confusion_matrix(
        test_true,
        test_pred,
        labels=[0, 1],
    )

    # ========================================================
    # Final print
    # ========================================================

    print("\n")
    print("=" * 100)
    print(
        f"SEED {seed} FINAL"
    )
    print("=" * 100)

    print(
        f"Best epoch        : "
        f"{best_epoch}"
    )

    print(
        f"Best Val Macro-F1 : "
        f"{best_val_macro_f1:.6f}"
    )

    print("\nValidation:")

    print(
        f"  Loss      : "
        f"{final_val['loss']:.6f}"
    )

    print(
        f"  Accuracy  : "
        f"{final_val['accuracy']:.6f}"
    )

    print(
        f"  Macro-F1  : "
        f"{final_val['macro_f1']:.6f}"
    )

    print(
        f"  Class1-F1 : "
        f"{final_val['class1_f1']:.6f}"
    )

    print("\nTest:")

    print(
        f"  Loss      : "
        f"{final_test['loss']:.6f}"
    )

    print(
        f"  Accuracy  : "
        f"{final_test['accuracy']:.6f}"
    )

    print(
        f"  Macro-F1  : "
        f"{final_test['macro_f1']:.6f}"
    )

    print(
        f"  Class1-F1 : "
        f"{final_test['class1_f1']:.6f}"
    )

    print(
        "\nValidation confusion matrix:"
    )

    print(val_cm)

    print(
        "\nTest confusion matrix:"
    )

    print(test_cm)

    print("=" * 100)

    return {
        "seed": int(seed),

        "best_epoch": int(
            best_epoch
        ),

        "best_val_macro_f1": float(
            best_val_macro_f1
        ),

        "val_loss": float(
            final_val["loss"]
        ),

        "val_accuracy": float(
            final_val["accuracy"]
        ),

        "val_macro_f1": float(
            final_val["macro_f1"]
        ),

        "val_class1_f1": float(
            final_val["class1_f1"]
        ),

        "test_loss": float(
            final_test["loss"]
        ),

        "test_accuracy": float(
            final_test["accuracy"]
        ),

        "test_macro_f1": float(
            final_test["macro_f1"]
        ),

        "test_class1_f1": float(
            final_test["class1_f1"]
        ),

        "val_confusion_matrix":
            val_cm.tolist(),

        "test_confusion_matrix":
            test_cm.tolist(),
    }


# ============================================================
# SUMMARY
# ============================================================

def summarize_results(
    results: List[Dict],
) -> Dict:

    metric_names = [
        "val_loss",
        "val_accuracy",
        "val_macro_f1",
        "val_class1_f1",
        "test_loss",
        "test_accuracy",
        "test_macro_f1",
        "test_class1_f1",
    ]

    summary = {}

    for metric in metric_names:

        values = np.asarray(
            [
                r[metric]
                for r in results
            ],
            dtype=np.float64,
        )

        summary[
            f"{metric}_mean"
        ] = float(
            np.mean(values)
        )

        summary[
            f"{metric}_std"
        ] = float(
            np.std(
                values,
                ddof=0,
            )
        )

        summary[
            f"{metric}_min"
        ] = float(
            np.min(values)
        )

        summary[
            f"{metric}_max"
        ] = float(
            np.max(values)
        )

    return summary


# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(
    experiment_name: str,
    subset_spec: Dict,

    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,

    scaler,

    categorical_cols: List[str],
    numerical_cols: List[str],

    seeds: List[int],
    device: torch.device,

    output_dir: Path,

    batch_size: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
):

    print("\n")
    print(
        "#" * 120
    )
    print(
        f"EXPERIMENT: "
        f"{experiment_name}"
    )
    print(
        "#" * 120
    )

    # ========================================================
    # Labels
    # ========================================================

    subset_mode = subset_spec[
        "mode"
    ]

    if subset_mode == "index":

        subset_indices = subset_spec[
            "indices"
        ]

        y_full = extract_target(
            train_df,
            "train.csv",
        )

        selected_y = (
            y_full[
                subset_indices
            ]
        )

    elif subset_mode == "materialized":

        subset_df = subset_spec[
            "df"
        ]

        selected_y = extract_target(
            subset_df,
            subset_spec["path"],
        )

    else:

        raise RuntimeError(
            f"Unknown subset mode: {subset_mode}"
        )

    values, counts = np.unique(
        selected_y,
        return_counts=True,
    )

    class_distribution = {
        int(v): int(c)
        for v, c
        in zip(values, counts)
    }

    print(
        f"Samples: "
        f"{len(selected_y)}"
    )

    print(
        f"Class distribution: "
        f"{class_distribution}"
    )

    print(
        f"Class-1 ratio: "
        f"{np.mean(selected_y == 1):.6f}"
    )

    # ========================================================
    # Prepare data
    # ========================================================

    if subset_mode == "index":

        (
            X_train_num,
            X_train_cat,
            y_train,

            X_val_num,
            X_val_cat,
            y_val,

            X_test_num,
            X_test_cat,
            y_test,

            cardinality_dict,
        ) = prepare_features(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_indices=subset_spec["indices"],
            scaler=scaler,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
        )

    else:

        (
            X_train_num,
            X_train_cat,
            y_train,

            X_val_num,
            X_val_cat,
            y_val,

            X_test_num,
            X_test_cat,
            y_test,

            cardinality_dict,
        ) = prepare_features_materialized(
            subset_df=subset_spec["df"],
            subset_path=subset_spec["path"],
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            scaler=scaler,
            categorical_cols=categorical_cols,
            numerical_cols=numerical_cols,
        )

    # ========================================================
    # IMPORTANT FIX
    #
    # Dict:
    #     used for logging/config
    #
    # List:
    #     used for nn.Embedding
    # ========================================================

    cardinality_list = [
        int(
            cardinality_dict[col]
        )
        for col in categorical_cols
    ]

    print(
        f"\nCategorical cardinalities "
        f"(dict): "
        f"{cardinality_dict}"
    )

    print(
        f"Categorical cardinalities "
        f"(list): "
        f"{cardinality_list}"
    )

    # ========================================================
    # Verify cardinalities
    # ========================================================

    if len(
        cardinality_list
    ) != 3:

        raise RuntimeError(
            f"\nExpected 3 categorical "
            f"cardinalities.\n"
            f"Got: {cardinality_list}"
        )

    # ========================================================
    # Shapes
    # ========================================================

    print(
        f"\nTrain numerical shape: "
        f"{X_train_num.shape}"
    )

    print(
        f"Train categorical shape: "
        f"{X_train_cat.shape}"
    )

    print(
        f"Validation numerical shape: "
        f"{X_val_num.shape}"
    )

    print(
        f"Validation categorical shape: "
        f"{X_val_cat.shape}"
    )

    print(
        f"Test numerical shape: "
        f"{X_test_num.shape}"
    )

    print(
        f"Test categorical shape: "
        f"{X_test_cat.shape}"
    )

    # ========================================================
    # Experiment folder
    # ========================================================

    experiment_dir = (
        output_dir
        / experiment_name
    )

    experiment_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Config
    # ========================================================

    config = {
        "experiment_name":
            str(experiment_name),

        "subset_mode":
            str(subset_mode),

        "subset_path":
            str(subset_spec["path"]),

        "num_samples":
            int(
                len(selected_y)
            ),

        "class0":
            int(
                np.sum(
                    selected_y == 0
                )
            ),

        "class1":
            int(
                np.sum(
                    selected_y == 1
                )
            ),

        "class1_ratio":
            float(
                np.mean(
                    selected_y == 1
                )
            ),

        "numerical_cols":
            list(numerical_cols),

        "categorical_cols":
            list(categorical_cols),

        "embedding_dim":
            int(EMBED_DIM),

        "input_dim":
            int(73),

        "hidden_dims":
            [256, 128, 64],

        "categorical_cardinalities":
            {
                str(k): int(v)
                for k, v
                in cardinality_dict.items()
            },

        "categorical_cardinality_list":
            [
                int(x)
                for x
                in cardinality_list
            ],

        "batch_size":
            int(batch_size),

        "epochs":
            int(epochs),

        "lr":
            float(lr),

        "weight_decay":
            float(weight_decay),

        "patience":
            int(patience),

        "seeds":
            [
                int(x)
                for x in seeds
            ],

        "device":
            str(device),
    }

    with open(
        experiment_dir
        / "config.json",
        "w",
    ) as f:

        json.dump(
            config,
            f,
            indent=2,
        )

    # ========================================================
    # Save the exact training subset used by the experiment
    # ========================================================

    if subset_mode == "index":

        pd.DataFrame(
            {
                "source_train_position":
                    subset_spec["indices"],
                "label":
                    selected_y,
            }
        ).to_csv(
            experiment_dir
            / "used_subset_indices.csv",
            index=False,
        )

    else:

        subset_spec["df"].to_csv(
            experiment_dir
            / "used_subset_materialized.csv",
            index=False,
        )

    # ========================================================
    # Train seeds
    # ========================================================

    seed_results = []

    for seed in seeds:

        checkpoint_path = (
            experiment_dir
            / f"best_seed{seed}.pt"
        )

        result = train_one_seed(
            X_train_num=X_train_num,
            X_train_cat=X_train_cat,
            y_train=y_train,

            X_val_num=X_val_num,
            X_val_cat=X_val_cat,
            y_val=y_val,

            X_test_num=X_test_num,
            X_test_cat=X_test_cat,
            y_test=y_test,

            categorical_cardinality_list=
                cardinality_list,

            seed=int(seed),

            device=device,

            batch_size=int(
                batch_size
            ),

            max_epochs=int(
                epochs
            ),

            lr=float(
                lr
            ),

            weight_decay=float(
                weight_decay
            ),

            patience=int(
                patience
            ),

            checkpoint_path=
                str(checkpoint_path),
        )

        seed_results.append(
            result
        )

    # ========================================================
    # Save seed results
    # ========================================================

    with open(
        experiment_dir
        / "seed_results.json",
        "w",
    ) as f:

        json.dump(
            seed_results,
            f,
            indent=2,
        )

    # ========================================================
    # Summary
    # ========================================================

    summary = summarize_results(
        seed_results
    )

    summary[
        "experiment_name"
    ] = str(
        experiment_name
    )

    summary[
        "num_samples"
    ] = int(
        len(selected_y)
    )

    summary[
        "class0"
    ] = int(
        np.sum(
            selected_y == 0
        )
    )

    summary[
        "class1"
    ] = int(
        np.sum(
            selected_y == 1
        )
    )

    summary[
        "class1_ratio"
    ] = float(
        np.mean(
            selected_y == 1
        )
    )

    with open(
        experiment_dir
        / "summary.json",
        "w",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )

    # ========================================================
    # Print experiment summary
    # ========================================================

    print("\n")
    print(
        "=" * 100
    )

    print(
        f"SUMMARY: "
        f"{experiment_name}"
    )

    print(
        "=" * 100
    )

    print(
        f"N = "
        f"{summary['num_samples']} "
        f"("
        f"{summary['class0']}"
        f" / "
        f"{summary['class1']}"
        f")"
    )

    print(
        f"Class-1 ratio = "
        f"{summary['class1_ratio']:.6f}"
    )

    print(
        f"Val Macro-F1 : "
        f"{summary['val_macro_f1_mean']:.6f} "
        f"+/- "
        f"{summary['val_macro_f1_std']:.6f}"
    )

    print(
        f"Val Class1-F1: "
        f"{summary['val_class1_f1_mean']:.6f} "
        f"+/- "
        f"{summary['val_class1_f1_std']:.6f}"
    )

    print(
        f"Val Accuracy : "
        f"{summary['val_accuracy_mean']:.6f} "
        f"+/- "
        f"{summary['val_accuracy_std']:.6f}"
    )

    print(
        f"Test Macro-F1: "
        f"{summary['test_macro_f1_mean']:.6f} "
        f"+/- "
        f"{summary['test_macro_f1_std']:.6f}"
    )

    print(
        f"Test Class1-F1: "
        f"{summary['test_class1_f1_mean']:.6f} "
        f"+/- "
        f"{summary['test_class1_f1_std']:.6f}"
    )

    print(
        "=" * 100
    )

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Step 5B V3 downstream MLP"
        )
    )

    # --------------------------------------------------------
    # Files
    # --------------------------------------------------------

    parser.add_argument(
        "--train",
        required=True,
    )

    parser.add_argument(
        "--val",
        required=True,
    )

    parser.add_argument(
        "--test",
        required=True,
    )

    parser.add_argument(
        "--teacher-dir",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    # --------------------------------------------------------
    # Subsets
    # --------------------------------------------------------

    parser.add_argument(
        "--subsets",
        nargs="*",
        default=[],
        help=(
            "NAME=PATH NAME=PATH ... "
            "(indexed OR materialized subset)"
        ),
    )

    parser.add_argument(
        "--include-full",
        action="store_true",
    )

    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LR,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=DEFAULT_WEIGHT_DECAY,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=DEFAULT_PATIENCE,
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
    )

    args = parser.parse_args()

    # ========================================================
    # Device
    # ========================================================

    if args.cpu:

        device = torch.device(
            "cpu"
        )

    else:

        device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    print(
        "=" * 100
    )

    print(
        "STEP 5B V3"
    )

    print(
        "=" * 100
    )

    print(
        f"PyTorch : "
        f"{torch.__version__}"
    )

    print(
        f"Device  : "
        f"{device}"
    )

    if device.type == "cuda":

        print(
            f"GPU     : "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"CUDA    : "
            f"{torch.version.cuda}"
        )

    print(
        "=" * 100
    )

    # ========================================================
    # Load data
    # ========================================================

    print(
        "\nLoading datasets..."
    )

    train_df = pd.read_csv(
        args.train
    )

    val_df = pd.read_csv(
        args.val
    )

    test_df = pd.read_csv(
        args.test
    )

    print(
        f"Train : "
        f"{train_df.shape}"
    )

    print(
        f"Val   : "
        f"{val_df.shape}"
    )

    print(
        f"Test  : "
        f"{test_df.shape}"
    )

    # ========================================================
    # Target
    # ========================================================

    train_target_col = (
        resolve_target_column(
            train_df,
            args.train,
        )
    )

    val_target_col = (
        resolve_target_column(
            val_df,
            args.val,
        )
    )

    test_target_col = (
        resolve_target_column(
            test_df,
            args.test,
        )
    )

    print(
        "\nTarget columns:"
    )

    print(
        f"  train = "
        f"{train_target_col}"
    )

    print(
        f"  val   = "
        f"{val_target_col}"
    )

    print(
        f"  test  = "
        f"{test_target_col}"
    )

    # ========================================================
    # Feature config
    # ========================================================

    input_dim = (
        len(NUMERICAL_COLS)
        + len(CATEGORICAL_COLS)
        * EMBED_DIM
    )

    print(
        "\nFeature configuration"
    )

    print(
        f"Numerical features: "
        f"{len(NUMERICAL_COLS)}"
    )

    print(
        f"Categorical features: "
        f"{len(CATEGORICAL_COLS)}"
    )

    print(
        f"Embedding dimension: "
        f"{EMBED_DIM}"
    )

    print(
        f"Total input dimension: "
        f"{input_dim}"
    )

    if input_dim != 73:

        raise RuntimeError(
            f"\nExpected model input "
            f"dimension 73.\n"
            f"Got {input_dim}."
        )

    # ========================================================
    # Scaler
    # ========================================================

    teacher_dir = Path(
        args.teacher_dir
    )

    scaler_path = (
        teacher_dir
        / "scaler.pkl"
    )

    scaler = load_scaler(
        scaler_path
    )

    verify_scaler(
        scaler,
        NUMERICAL_COLS,
    )

    # ========================================================
    # Output directory
    # ========================================================

    output_dir = Path(
        args.output
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # Experiments
    # ========================================================

    print(
        "\nSubset mode:"
    )

    print(
        "  Indexed subset: "
        "source_train_index/source_train_position"
    )

    print(
        "  Materialized subset: "
        "actual 29-column training rows"
    )

    print(
        "  Materialized mode is required for synthetic "
        "samples not present in train.csv."
    )

    experiments = {}

    # --------------------------------------------------------
    # Full
    # --------------------------------------------------------

    if args.include_full:

        experiments[
            "full"
        ] = {
            "mode": "index",
            "path": args.train,
            "indices": np.arange(
                len(train_df),
                dtype=np.int64,
            ),
            "df": None,
        }

    # --------------------------------------------------------
    # Subsets
    # --------------------------------------------------------

    for item in args.subsets:

        if "=" not in item:

            raise ValueError(
                f"\nInvalid subset argument:\n"
                f"{item}\n\n"
                f"Expected:\n"
                f"NAME=PATH"
            )

        name, subset_path = (
            item.split(
                "=",
                1,
            )
        )

        name = name.strip()
        subset_path = subset_path.strip()

        if not name:

            raise ValueError(
                f"\nEmpty experiment name:\n"
                f"{item}"
            )

        if not subset_path:

            raise ValueError(
                f"\nEmpty subset path:\n"
                f"{item}"
            )

        if not os.path.isfile(
            subset_path
        ):

            raise FileNotFoundError(
                f"\nSubset file not found:\n"
                f"{subset_path}"
            )

        subset_spec = load_subset_spec(
            subset_path=subset_path,
            train_df=train_df,
            numerical_cols=NUMERICAL_COLS,
            categorical_cols=CATEGORICAL_COLS,
        )

        if name in experiments:

            raise RuntimeError(
                f"\nDuplicate experiment name:\n"
                f"{name}"
            )

        experiments[
            name
        ] = subset_spec

    if not experiments:

        raise RuntimeError(
            "\nNo experiments specified.\n\n"
            "Use:\n"
            "  --include-full\n"
            "and/or\n"
            "  --subsets NAME=PATH ..."
        )

    # ========================================================
    # Run
    # ========================================================

    all_summaries = {}

    for (
        experiment_name,
        subset_spec,
    ) in experiments.items():

        summary = run_experiment(
            experiment_name=
                experiment_name,

            subset_spec=
                subset_spec,

            train_df=
                train_df,

            val_df=
                val_df,

            test_df=
                test_df,

            scaler=
                scaler,

            categorical_cols=
                CATEGORICAL_COLS,

            numerical_cols=
                NUMERICAL_COLS,

            seeds=
                args.seeds,

            device=
                device,

            output_dir=
                output_dir,

            batch_size=
                args.batch_size,

            epochs=
                args.epochs,

            lr=
                args.lr,

            weight_decay=
                args.weight_decay,

            patience=
                args.patience,
        )

        all_summaries[
            experiment_name
        ] = summary

    # ========================================================
    # Save all results
    # ========================================================

    all_results_path = (
        output_dir
        / "all_results.json"
    )

    with open(
        all_results_path,
        "w",
    ) as f:

        json.dump(
            all_summaries,
            f,
            indent=2,
        )

    # ========================================================
    # Comparison CSV
    # ========================================================

    comparison_rows = []

    for (
        name,
        summary,
    ) in all_summaries.items():

        comparison_rows.append(
            {
                "experiment":
                    name,

                "num_samples":
                    summary[
                        "num_samples"
                    ],

                "class0":
                    summary[
                        "class0"
                    ],

                "class1":
                    summary[
                        "class1"
                    ],

                "class1_ratio":
                    summary[
                        "class1_ratio"
                    ],

                "val_macro_f1_mean":
                    summary[
                        "val_macro_f1_mean"
                    ],

                "val_macro_f1_std":
                    summary[
                        "val_macro_f1_std"
                    ],

                "val_class1_f1_mean":
                    summary[
                        "val_class1_f1_mean"
                    ],

                "val_class1_f1_std":
                    summary[
                        "val_class1_f1_std"
                    ],

                "val_accuracy_mean":
                    summary[
                        "val_accuracy_mean"
                    ],

                "val_accuracy_std":
                    summary[
                        "val_accuracy_std"
                    ],

                "test_macro_f1_mean":
                    summary[
                        "test_macro_f1_mean"
                    ],

                "test_macro_f1_std":
                    summary[
                        "test_macro_f1_std"
                    ],

                "test_class1_f1_mean":
                    summary[
                        "test_class1_f1_mean"
                    ],

                "test_class1_f1_std":
                    summary[
                        "test_class1_f1_std"
                    ],
            }
        )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    comparison_path = (
        output_dir
        / "comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    # ========================================================
    # Final comparison
    # ========================================================

    print("\n\n")

    print(
        "#" * 120
    )

    print(
        "FINAL COMPARISON"
    )

    print(
        "#" * 120
    )

    columns = [
        "experiment",
        "num_samples",
        "class0",
        "class1",
        "class1_ratio",
        "val_macro_f1_mean",
        "val_macro_f1_std",
        "val_class1_f1_mean",
        "val_class1_f1_std",
        "val_accuracy_mean",
        "val_accuracy_std",
        "test_macro_f1_mean",
        "test_macro_f1_std",
        "test_class1_f1_mean",
        "test_class1_f1_std",
    ]

    print(
        comparison_df[
            columns
        ].to_string(
            index=False,
            float_format=(
                lambda x:
                f"{x:.6f}"
            ),
        )
    )

    print(
        "\nAll results:"
    )

    print(
        all_results_path
    )

    print(
        "\nComparison:"
    )

    print(
        comparison_path
    )

    print(
        "\n"
        + "#" * 120
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()