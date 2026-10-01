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

from sklearn.cluster import MiniBatchKMeans


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
# Must be identical to train_teacher.py
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

            nn.Dropout(dropout),

            nn.Linear(
                256,
                128
            ),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                128,
                representation_dim
            ),

            nn.ReLU()
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

def load_metadata(
    teacher_dir
):

    teacher_dir = Path(
        teacher_dir
    )

    with open(
        teacher_dir / "preprocessor.json",
        "r"
    ) as f:

        metadata = json.load(f)

    scaler = joblib.load(
        teacher_dir / "scaler.pkl"
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

    numeric_cols = metadata[
        "numeric_cols"
    ]

    categorical_cols = metadata[
        "categorical_cols"
    ]

    numeric_medians = metadata[
        "numeric_medians"
    ]

    category_maps = metadata[
        "category_maps"
    ]

    label_map = metadata[
        "label_map"
    ]

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
                float(
                    numeric_medians[col]
                )
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
    # Labels
    # --------------------------------------------------------

    y = np.array(
        [
            label_map[str(v)]
            for v in df[LABEL_COL]
        ],
        dtype=np.int64
    )

    return X_num, X_cat, y


# ============================================================
# EXTRACT TEACHER INPUT REPRESENTATION
# ============================================================

@torch.no_grad()
def extract_input_representation(
    model,
    X_num,
    X_cat,
    device,
    batch_size=4096
):

    model.eval()

    N = len(X_num)

    representations = []

    for start in range(
        0,
        N,
        batch_size
    ):

        end = min(
            start + batch_size,
            N
        )

        X_num_batch = torch.from_numpy(
            X_num[start:end]
        ).to(
            device
        )

        X_cat_batch = torch.from_numpy(
            X_cat[start:end]
        ).to(
            device
        )

        # Numerical part
        num_part = (
            X_num_batch
        )

        # Categorical embedding parts
        embedding_parts = []

        for i, embedding in enumerate(
            model.embeddings
        ):

            emb = embedding(
                X_cat_batch[:, i]
            )

            embedding_parts.append(
                emb
            )

        # Concatenate numerical + embeddings
        Z = torch.cat(
            [
                num_part,
                *embedding_parts
            ],
            dim=1
        )

        representations.append(
            Z.cpu().numpy().astype(
                np.float32
            )
        )

    return np.concatenate(
        representations,
        axis=0
    )


# ============================================================
# STANDARDIZE EMBEDDING BLOCKS
# ============================================================

def standardize_embedding_blocks(
    Z,
    metadata
):

    numeric_cols = metadata[
        "numeric_cols"
    ]

    categorical_cols = metadata[
        "categorical_cols"
    ]

    category_maps = metadata[
        "category_maps"
    ]

    num_numeric = len(
        numeric_cols
    )

    Z_out = Z.copy()

    feature_slices = {}

    # Numerical features
    for i, col in enumerate(
        numeric_cols
    ):

        feature_slices[col] = (
            i,
            i + 1
        )

    current = num_numeric

    # Each embedding block
    for col in categorical_cols:

        cardinality = (
            len(
                category_maps[col]
            ) + 1
        )

        # Actual embedding dimension
        # inferred from the representation.
        #
        # For this teacher:
        # embedding dim = 16 unless cardinality is tiny.
        #
        # We find block dimension by metadata
        # outside; here we use the expected 16.
        #
        # This function receives exact slices
        # separately below.
        pass

    return Z_out


# ============================================================
# CREATE REPRESENTATION BLOCK SLICES
# ============================================================

def get_feature_slices(
    model,
    metadata
):

    numeric_cols = metadata[
        "numeric_cols"
    ]

    categorical_cols = metadata[
        "categorical_cols"
    ]

    slices = {}

    # Numerical
    for i, col in enumerate(
        numeric_cols
    ):

        slices[col] = (
            i,
            i + 1
        )

    current = len(
        numeric_cols
    )

    # Categorical embeddings
    for col, embedding in zip(
        categorical_cols,
        model.embeddings
    ):

        dim = embedding.embedding_dim

        slices[col] = (
            current,
            current + dim
        )

        current += dim

    return slices, current


# ============================================================
# NORMALIZE EMBEDDING BLOCKS
# ============================================================

def normalize_representation_blocks(
    Z,
    model,
    metadata
):

    Z = Z.copy()

    slices, total_dim = (
        get_feature_slices(
            model,
            metadata
        )
    )

    assert (
        Z.shape[1] == total_dim
    ), (
        f"Representation dimension "
        f"{Z.shape[1]} != expected "
        f"{total_dim}"
    )

    # Numerical are already standardized.
    # Only normalize categorical embedding blocks.
    for col in metadata[
        "categorical_cols"
    ]:

        start, end = slices[col]

        block = Z[:, start:end]

        mean = block.mean(
            axis=0,
            keepdims=True
        )

        std = block.std(
            axis=0,
            keepdims=True
        )

        std = np.maximum(
            std,
            1e-6
        )

        Z[:, start:end] = (
            block - mean
        ) / std

    return Z


# ============================================================
# LOAD / BUILD FEATURE WEIGHTS
# ============================================================

def normalize_positive(
    values
):

    values = np.asarray(
        values,
        dtype=np.float64
    )

    values = np.maximum(
        values,
        0.0
    )

    total = values.sum()

    if total <= 0:
        return np.ones_like(values) / len(values)

    return values / total


def build_feature_weights(
    importance_csv,
    feature_names,
    alpha_global,
    alpha_class
):

    df = pd.read_csv(
        importance_csv
    )

    df = df.set_index(
        "feature"
    )

    # --------------------------------------------------------
    # Use normalized columns if available.
    # Otherwise normalize positive means.
    # --------------------------------------------------------

    if (
        "global_normalized" in df.columns
    ):

        global_imp = np.array(
            [
                df.loc[f, "global_normalized"]
                for f in feature_names
            ],
            dtype=np.float64
        )

    else:

        global_imp = normalize_positive(
            [
                df.loc[f, "global_mean"]
                for f in feature_names
            ]
        )

    class0_imp = normalize_positive(
        [
            df.loc[f, "class_0_mean"]
            for f in feature_names
        ]
    )

    class1_imp = normalize_positive(
        [
            df.loc[f, "class_1_mean"]
            for f in feature_names
        ]
    )

    # Global is already normalized.
    global_imp = normalize_positive(
        global_imp
    )

    # --------------------------------------------------------
    # Class-specific mixtures
    # --------------------------------------------------------

    W0 = (
        alpha_global * global_imp
        +
        alpha_class * class0_imp
    )

    W1 = (
        alpha_global * global_imp
        +
        alpha_class * class1_imp
    )

    W0 = normalize_positive(
        W0
    )

    W1 = normalize_positive(
        W1
    )

    weights_df = pd.DataFrame(
        {
            "feature": feature_names,
            "global": global_imp,
            "class0": class0_imp,
            "class1": class1_imp,
            "weight_class0": W0,
            "weight_class1": W1,
        }
    )

    return (
        weights_df,
        W0,
        W1
    )


# ============================================================
# APPLY FEATURE WEIGHTS
# ============================================================

def apply_block_weights(
    Z,
    model,
    metadata,
    weights,
):

    Z_weighted = Z.copy()

    feature_names = (
        metadata["numeric_cols"]
        +
        metadata["categorical_cols"]
    )

    slices, total_dim = (
        get_feature_slices(
            model,
            metadata
        )
    )

    assert len(weights) == len(
        feature_names
    )

    # --------------------------------------------------------
    # Weight every feature block by sqrt(weight)
    # --------------------------------------------------------

    for idx, feature in enumerate(
        feature_names
    ):

        start, end = slices[
            feature
        ]

        scale = np.sqrt(
            weights[idx]
        )

        Z_weighted[:, start:end] *= (
            scale
        )

    return Z_weighted


# ============================================================
# KMEANS + CANDIDATE EXTRACTION
# ============================================================

def run_kmeans_candidates(
    Z,
    global_indices,
    n_clusters,
    candidates_per_cluster,
    seed,
    batch_size,
    max_iter
):

    N = len(Z)

    n_clusters = min(
        int(n_clusters),
        N
    )

    print(
        f"  samples      : {N}"
    )

    print(
        f"  clusters     : {n_clusters}"
    )

    print(
        f"  per cluster  : "
        f"{candidates_per_cluster}"
    )

    if n_clusters <= 0:
        return pd.DataFrame()

    kmeans = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=10,
        random_state=seed,
        reassignment_ratio=0.01,
    )

    cluster_labels = (
        kmeans.fit_predict(Z)
    )

    centers = kmeans.cluster_centers_

    candidates = []

    # --------------------------------------------------------
    # Process each cluster separately
    # Avoid creating N x K distance matrix.
    # --------------------------------------------------------

    for cluster_id in range(
        n_clusters
    ):

        member_mask = (
            cluster_labels
            == cluster_id
        )

        member_positions = np.flatnonzero(
            member_mask
        )

        if len(member_positions) == 0:
            continue

        cluster_points = Z[
            member_positions
        ]

        center = centers[
            cluster_id
        ]

        distances = np.sum(
            (
                cluster_points
                - center
            ) ** 2,
            axis=1
        )

        # Nearest samples to centroid
        order = np.argsort(
            distances
        )

        take = min(
            candidates_per_cluster,
            len(order)
        )

        selected_positions = (
            member_positions[
                order[:take]
            ]
        )

        selected_distances = (
            distances[
                order[:take]
            ]
        )

        for pos, dist in zip(
            selected_positions,
            selected_distances
        ):

            candidates.append(
                {
                    "global_index":
                        int(global_indices[pos]),

                    "cluster":
                        int(cluster_id),

                    "centroid_distance":
                        float(dist),

                    "candidate_rank_in_cluster":
                        int(
                            np.where(
                                selected_positions
                                == pos
                            )[0][0]
                        ),
                }
            )

    return pd.DataFrame(
        candidates
    )


# ============================================================
# MAIN
# ============================================================

def main(args):

    set_seed(
        args.seed
    )

    teacher_dir = Path(
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
    print("STEP 3 - CLASS-SPECIFIC FEATURE-WEIGHTED KMEANS")
    print("=" * 80)

    print(
        f"Device: {device}"
    )

    # ========================================================
    # 1. LOAD RAW TRAIN
    # ========================================================

    print(
        "\n[1/8] Loading RAW train data"
    )

    train_df = pd.read_csv(
        args.train
    )

    print(
        f"Train shape: "
        f"{train_df.shape}"
    )

    y_raw = train_df[
        LABEL_COL
    ].values

    # ========================================================
    # 2. LOAD PREPROCESSING
    # ========================================================

    print(
        "\n[2/8] Loading preprocessing"
    )

    metadata, scaler = (
        load_metadata(
            teacher_dir
        )
    )

    numeric_cols = metadata[
        "numeric_cols"
    ]

    categorical_cols = metadata[
        "categorical_cols"
    ]

    feature_names = (
        numeric_cols
        +
        categorical_cols
    )

    print(
        "Features:"
    )

    for feature in feature_names:
        print(
            f"  {feature}"
        )

    # ========================================================
    # 3. PREPROCESS
    # ========================================================

    print(
        "\n[3/8] Transforming train data"
    )

    X_num, X_cat, y = (
        transform_dataframe(
            train_df,
            metadata,
            scaler
        )
    )

    print(
        f"X_num: {X_num.shape}"
    )

    print(
        f"X_cat: {X_cat.shape}"
    )

    # ========================================================
    # 4. LOAD TEACHER
    # ========================================================

    print(
        "\n[4/8] Loading teacher"
    )

    category_cardinalities = [
        len(
            metadata[
                "category_maps"
            ][col]
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
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(
        teacher_dir / "teacher_best.pt",
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        f"Teacher epoch: "
        f"{checkpoint.get('epoch', 'unknown')}"
    )

    # ========================================================
    # 5. EXTRACT 73-D REPRESENTATION
    # ========================================================

    print(
        "\n[5/8] Extracting teacher input representation"
    )

    Z = extract_input_representation(
        model=model,
        X_num=X_num,
        X_cat=X_cat,
        device=device,
        batch_size=args.batch_size
    )

    print(
        f"Raw teacher representation: "
        f"{Z.shape}"
    )

    # --------------------------------------------------------
    # Normalize embedding blocks
    # --------------------------------------------------------

    Z = normalize_representation_blocks(
        Z,
        model,
        metadata
    )

    print(
        f"Normalized representation: "
        f"{Z.shape}"
    )

    np.save(
        result_dir /
        "teacher_input_representation.npy",
        Z
    )

    # ========================================================
    # 6. BUILD FEATURE WEIGHTS
    # ========================================================

    print(
        "\n[6/8] Building class-specific feature weights"
    )

    (
        weights_df,
        W0,
        W1
    ) = build_feature_weights(
        importance_csv=args.importance_csv,
        feature_names=feature_names,
        alpha_global=args.global_weight,
        alpha_class=args.class_weight
    )

    weights_df.to_csv(
        result_dir /
        "class_specific_feature_weights.csv",
        index=False
    )

    print(
        "\nClass 0 weights:"
    )

    print(
        weights_df[
            [
                "feature",
                "weight_class0"
            ]
        ]
        .sort_values(
            "weight_class0",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    print(
        "\nClass 1 weights:"
    )

    print(
        weights_df[
            [
                "feature",
                "weight_class1"
            ]
        ]
        .sort_values(
            "weight_class1",
            ascending=False
        )
        .to_string(
            index=False
        )
    )

    # ========================================================
    # APPLY CLASS-SPECIFIC WEIGHTS
    # ========================================================

    print(
        "\nApplying class-specific geometry..."
    )

    Z0 = apply_block_weights(
        Z,
        model,
        metadata,
        W0
    )

    Z1 = apply_block_weights(
        Z,
        model,
        metadata,
        W1
    )

    # Save weighted representations
    np.save(
        result_dir /
        "weighted_representation_class0.npy",
        Z0.astype(np.float32)
    )

    np.save(
        result_dir /
        "weighted_representation_class1.npy",
        Z1.astype(np.float32)
    )

    # ========================================================
    # 7. CANDIDATE BUDGET
    # ========================================================

    print(
        "\n[7/8] Computing candidate budgets"
    )

    train_indices = np.arange(
        len(y)
    )

    class0_indices = (
        train_indices[y == 0]
    )

    class1_indices = (
        train_indices[y == 1]
    )

    N = len(y)

    final_budget = (
        args.final_ratio * N
    )

    final_budget = max(
        1,
        int(round(final_budget))
    )

    candidate_budget = int(
        round(
            final_budget
            * args.candidate_multiplier
        )
    )

    # Candidate class-1 fraction
    candidate_k1 = int(
        round(
            candidate_budget
            * args.candidate_class1_ratio
        )
    )

    candidate_k0 = (
        candidate_budget
        - candidate_k1
    )

    candidate_k0 = min(
        candidate_k0,
        len(class0_indices)
    )

    candidate_k1 = min(
        candidate_k1,
        len(class1_indices)
    )

    # Safety
    candidate_k0 = max(
        1,
        candidate_k0
    )

    candidate_k1 = max(
        1,
        candidate_k1
    )

    print(
        f"Train samples        : {N}"
    )

    print(
        f"Final budget         : "
        f"{final_budget}"
    )

    print(
        f"Candidate budget     : "
        f"{candidate_budget}"
    )

    print(
        f"Candidate class 0   : "
        f"{candidate_k0}"
    )

    print(
        f"Candidate class 1   : "
        f"{candidate_k1}"
    )

    # --------------------------------------------------------
    # Number of clusters
    #
    # We generate candidates_per_cluster points
    # from each centroid.
    # --------------------------------------------------------

    K0 = int(
        np.ceil(
            candidate_k0
            / args.candidates_per_cluster
        )
    )

    K1 = int(
        np.ceil(
            candidate_k1
            / args.candidates_per_cluster
        )
    )

    K0 = min(
        K0,
        len(class0_indices)
    )

    K1 = min(
        K1,
        len(class1_indices)
    )

    print(
        f"\nKMeans class 0 clusters: "
        f"{K0}"
    )

    print(
        f"KMeans class 1 clusters: "
        f"{K1}"
    )

    # ========================================================
    # 8. RUN KMEANS
    # ========================================================

    print(
        "\n[8/8] Running class-specific KMeans"
    )

    # --------------------------------------------------------
    # CLASS 0
    # --------------------------------------------------------

    print(
        "\n========== CLASS 0 =========="
    )

    Z0_class = Z0[
        class0_indices
    ]

    candidates0 = (
        run_kmeans_candidates(
            Z=Z0_class,
            global_indices=class0_indices,
            n_clusters=K0,
            candidates_per_cluster=(
                args.candidates_per_cluster
            ),
            seed=args.seed,
            batch_size=args.kmeans_batch_size,
            max_iter=args.kmeans_max_iter
        )
    )

    candidates0["class"] = 0

    # --------------------------------------------------------
    # CLASS 1
    # --------------------------------------------------------

    print(
        "\n========== CLASS 1 =========="
    )

    Z1_class = Z1[
        class1_indices
    ]

    candidates1 = (
        run_kmeans_candidates(
            Z=Z1_class,
            global_indices=class1_indices,
            n_clusters=K1,
            candidates_per_cluster=(
                args.candidates_per_cluster
            ),
            seed=args.seed + 1,
            batch_size=args.kmeans_batch_size,
            max_iter=args.kmeans_max_iter
        )
    )

    candidates1["class"] = 1

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    candidates = pd.concat(
        [
            candidates0,
            candidates1
        ],
        ignore_index=True
    )

    # Remove duplicates just in case
    candidates = (
        candidates
        .drop_duplicates(
            subset=["global_index"]
        )
        .reset_index(drop=True)
    )

    # Attach original labels
    candidates["label"] = (
        y[
            candidates["global_index"]
            .values
        ]
    )

    # Sort by class / cluster
    candidates = candidates.sort_values(
        [
            "class",
            "cluster",
            "centroid_distance"
        ]
    ).reset_index(
        drop=True
    )

    candidates.to_csv(
        result_dir /
        "candidate_pool.csv",
        index=False
    )

    # --------------------------------------------------------
    # Candidate indices
    # --------------------------------------------------------

    np.save(
        result_dir /
        "candidate_indices.npy",
        candidates[
            "global_index"
        ].values.astype(
            np.int64
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = {

        "num_train_samples":
            int(N),

        "final_budget":
            int(final_budget),

        "candidate_budget_target":
            int(candidate_budget),

        "candidate_count_actual":
            int(len(candidates)),

        "candidate_count_class0":
            int(
                (candidates["class"] == 0)
                .sum()
            ),

        "candidate_count_class1":
            int(
                (candidates["class"] == 1)
                .sum()
            ),

        "candidate_class1_ratio":
            args.candidate_class1_ratio,

        "candidate_multiplier":
            args.candidate_multiplier,

        "candidates_per_cluster":
            args.candidates_per_cluster,

        "KMeans_clusters_class0":
            int(K0),

        "KMeans_clusters_class1":
            int(K1),

        "global_weight_ratio":
            args.global_weight,

        "class_specific_weight_ratio":
            args.class_weight,

        "seed":
            args.seed
    }

    with open(
        result_dir /
        "step3_summary.json",
        "w"
    ) as f:

        json.dump(
            summary,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # FINAL LOG
    # --------------------------------------------------------

    print(
        "\n" + "=" * 80
    )

    print(
        "STEP 3 FINISHED"
    )

    print(
        "=" * 80
    )

    print(
        f"\nFinal budget:"
        f" {final_budget}"
    )

    print(
        f"Candidate pool:"
        f" {len(candidates)}"
    )

    print(
        "\nCandidate class distribution:"
    )

    print(
        candidates[
            "class"
        ].value_counts()
        .sort_index()
    )

    print(
        "\nSaved:"
    )

    print(
        result_dir /
        "class_specific_feature_weights.csv"
    )

    print(
        result_dir /
        "teacher_input_representation.npy"
    )

    print(
        result_dir /
        "weighted_representation_class0.npy"
    )

    print(
        result_dir /
        "weighted_representation_class1.npy"
    )

    print(
        result_dir /
        "candidate_pool.csv"
    )

    print(
        result_dir /
        "candidate_indices.npy"
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train",
        required=True,
        help="RAW train CSV"
    )

    parser.add_argument(
        "--teacher-dir",
        required=True,
        help="Teacher output directory"
    )

    parser.add_argument(
        "--importance-csv",
        required=True,
        help="feature_importance.csv"
    )

    parser.add_argument(
        "--result-dir",
        required=True
    )

    # --------------------------------------------------------
    # Final budget
    # --------------------------------------------------------

    parser.add_argument(
        "--final-ratio",
        type=float,
        default=0.01
    )

    # --------------------------------------------------------
    # Candidate pool
    # --------------------------------------------------------

    parser.add_argument(
        "--candidate-multiplier",
        type=float,
        default=4.0,
        help="Candidate pool / final budget"
    )

    parser.add_argument(
        "--candidate-class1-ratio",
        type=float,
        default=0.30,
        help="Fraction of candidate pool reserved for class 1"
    )

    parser.add_argument(
        "--candidates-per-cluster",
        type=int,
        default=2
    )

    # --------------------------------------------------------
    # Feature importance mixing
    # --------------------------------------------------------

    parser.add_argument(
        "--global-weight",
        type=float,
        default=0.7
    )

    parser.add_argument(
        "--class-weight",
        type=float,
        default=0.3
    )

    # --------------------------------------------------------
    # Training / KMeans
    # --------------------------------------------------------

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096
    )

    parser.add_argument(
        "--kmeans-batch-size",
        type=int,
        default=4096
    )

    parser.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=200
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    if not np.isclose(
        args.global_weight
        + args.class_weight,
        1.0
    ):
        raise ValueError(
            "--global-weight + "
            "--class-weight must equal 1.0"
        )

    if not (
        0.0
        < args.candidate_class1_ratio
        < 1.0
    ):
        raise ValueError(
            "candidate-class1-ratio "
            "must be between 0 and 1."
        )

    main(args)