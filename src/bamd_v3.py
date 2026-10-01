#!/usr/bin/env python3
"""BAMD-v3: joint class selection with bounded boundary support.

Protocol
--------
- Use the canonical 2,532-sample Step-3 candidate pool.
- Select BOTH classes from their candidate pools:
    Class 0: 1,772 candidates -> 389 samples
    Class 1:   760 candidates -> 244 samples
- Keep one selected sample per KMeans cluster, as in P2/BAMD-v2.
- Preserve the P2-compatible score structure:
      S = 0.55 * R + 0.30 * B + 0.15 * D_mix
- Boundary support is bounded:
      U_i = min(p_i0, p_i1)
      L_i = d_same / (d_same + d_opp + eps)
      B_i = U_i * L_i
- Mixed diversity is:
      D_mix = alpha * cosine_distance(latent)
            + (1-alpha) * PFI-weighted categorical Hamming distance
- Categorical weights are class-specific normalized PFI weights:
      class_0_normalized for C0, class_1_normalized for C1.

This is the symmetric/joint counterpart of BAMD-v2 while preserving
candidate pool, class budget, KMeans, one-sample-per-cluster, and downstream
protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors

CATEGORICAL_COLS = ["node_id", "parent_id", "rpl_ver"]
LABEL_CANDIDATES = ["label", "target"]

W_REP = 0.55
W_BOUNDARY = 0.30
W_DIVERSITY = 0.15


def rank_normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    if n <= 1:
        return np.ones_like(values, dtype=np.float64)
    order = np.argsort(np.argsort(values, kind="mergesort"), kind="mergesort")
    return order.astype(np.float64) / float(n - 1)


def load_label_col(df: pd.DataFrame) -> str:
    for col in LABEL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(f"Missing label column; expected one of {LABEL_CANDIDATES}")


def load_pfi_weights(path: Path) -> tuple[dict[str, float], dict[str, float]]:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"PFI file is empty: {path}")

    feature_col = next((c for c in ["feature", "feature_name", "Feature", "name", "column"] if c in df.columns), None)
    if feature_col is None:
        raise ValueError(f"Cannot find feature-name column in {path}. Columns: {list(df.columns)}")

    c0_col = next((c for c in ["class_0_normalized", "class_0_mean", "global_normalized", "global_mean"] if c in df.columns), None)
    c1_col = next((c for c in ["class_1_normalized", "class_1_mean", "global_normalized", "global_mean"] if c in df.columns), None)
    if c0_col is None or c1_col is None:
        raise ValueError(
            f"PFI file needs class-specific normalized/mean columns. Columns: {list(df.columns)}"
        )

    raw0: dict[str, float] = {}
    raw1: dict[str, float] = {}
    for _, row in df.iterrows():
        name = str(row[feature_col])
        try:
            raw0[name] = float(row[c0_col])
        except (TypeError, ValueError):
            pass
        try:
            raw1[name] = float(row[c1_col])
        except (TypeError, ValueError):
            pass

    def normalize(raw: dict[str, float], col_name: str) -> dict[str, float]:
        vals = {c: max(raw.get(c, 0.0), 0.0) for c in CATEGORICAL_COLS}
        total = sum(vals.values())
        if total <= 0:
            raise ValueError(f"All PFI weights are non-positive for {col_name}: {vals}")
        return {c: vals[c] / total for c in CATEGORICAL_COLS}

    w0 = normalize(raw0, c0_col)
    w1 = normalize(raw1, c1_col)
    print(f"Using PFI columns: C0={c0_col}, C1={c1_col}", flush=True)
    return w0, w1


def factorize_categories(df: pd.DataFrame, positions: np.ndarray) -> np.ndarray:
    blocks = []
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            raise ValueError(f"Missing categorical column: {col}")
        codes, _ = pd.factorize(df[col], sort=True)
        blocks.append(codes[positions].astype(np.int64))
    return np.column_stack(blocks)


def bounded_boundary_support(
    boundary_rep: np.ndarray,
    labels: np.ndarray,
    candidate_positions: np.ndarray,
    *,
    k_same: int,
    k_opp: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return bounded support L = d_same/(d_same+d_opp), same mean, opp mean."""
    X = np.asarray(boundary_rep, dtype=np.float32)
    if X.shape[0] != len(labels):
        raise ValueError("boundary_rep rows must equal train rows")

    cls = int(labels[candidate_positions[0]])
    same_pos = np.flatnonzero(labels == cls)
    opp_pos = np.flatnonzero(labels != cls)
    if len(same_pos) <= k_same:
        raise ValueError(f"Not enough same-class samples for k_same={k_same}, class={cls}")
    if len(opp_pos) < k_opp:
        raise ValueError(f"Not enough opposite-class samples for k_opp={k_opp}, class={cls}")

    Xq = X[candidate_positions]
    Xs = X[same_pos]
    Xo = X[opp_pos]

    same_nn = NearestNeighbors(n_neighbors=k_same + 1, metric="euclidean")
    same_nn.fit(Xs)
    same_dist, same_idx = same_nn.kneighbors(Xq, return_distance=True)

    pos_to_local = {int(p): i for i, p in enumerate(same_pos)}
    same_means = np.empty(len(candidate_positions), dtype=np.float64)
    for i, gp in enumerate(candidate_positions):
        self_local = pos_to_local.get(int(gp))
        vals = same_dist[i]
        if self_local is not None:
            hit = np.where(same_idx[i] == self_local)[0]
            if hit.size:
                vals = np.delete(vals, hit[0])
        vals = vals[:k_same]
        same_means[i] = float(np.mean(vals))

    opp_nn = NearestNeighbors(n_neighbors=k_opp, metric="euclidean")
    opp_nn.fit(Xo)
    opp_dist, _ = opp_nn.kneighbors(Xq, return_distance=True)
    opp_means = opp_dist.mean(axis=1)

    support = same_means / (same_means + opp_means + 1e-8)
    return support, same_means, opp_means


def cosine_distance_all(Xn: np.ndarray, best_idx: int) -> np.ndarray:
    sim = np.clip(Xn @ Xn[best_idx], -1.0, 1.0)
    return (1.0 - sim) / 2.0


def select_class(
    *,
    class_id: int,
    candidate_positions: np.ndarray,
    weighted_rep: np.ndarray,
    boundary_rep: np.ndarray,
    probs: np.ndarray,
    labels: np.ndarray,
    cat_codes: np.ndarray,
    cat_weights: np.ndarray,
    budget: int,
    alpha: float,
    k_same: int,
    k_opp: int,
    seed: int,
    n_init: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    if len(candidate_positions) < budget:
        raise ValueError(f"Class {class_id}: budget {budget} exceeds candidates {len(candidate_positions)}")

    X = np.asarray(weighted_rep[candidate_positions], dtype=np.float64)
    if X.ndim != 2:
        raise ValueError(f"Class {class_id}: representation must be 2-D, got {X.shape}")

    # Teacher uncertainty.
    cand_probs = np.asarray(probs[candidate_positions], dtype=np.float64)
    uncertainty = np.min(cand_probs, axis=1)

    # Bounded boundary support.
    support, same_mean, opp_mean = bounded_boundary_support(
        boundary_rep,
        labels,
        candidate_positions,
        k_same=k_same,
        k_opp=k_opp,
    )
    boundary_raw = uncertainty * support
    boundary_rank = rank_normalize(boundary_raw)

    # Class-specific KMeans: preserve P2's one-sample-per-cluster structure.
    km = KMeans(
        n_clusters=budget,
        random_state=seed,
        n_init=n_init,
        algorithm="lloyd",
    )
    cluster_labels = km.fit_predict(X)
    centers = km.cluster_centers_

    cluster_info: dict[int, dict[str, np.ndarray]] = {}
    for cid in range(budget):
        local = np.flatnonzero(cluster_labels == cid)
        if local.size == 0:
            raise RuntimeError(f"Class {class_id}: empty KMeans cluster {cid}")
        dist = np.linalg.norm(X[local] - centers[cid], axis=1)
        rep_rank = 1.0 - rank_normalize(dist)
        cluster_info[cid] = {"local": local, "dist": dist, "rep_rank": rep_rank}

    norms = np.linalg.norm(X, axis=1, keepdims=True)
    Xn = X / np.where(norms > 1e-12, norms, 1.0)

    min_mix = np.full(len(candidate_positions), np.inf, dtype=np.float64)
    available = np.ones(len(candidate_positions), dtype=bool)
    selected: list[int] = []
    selected_clusters: set[int] = set()
    audit_rows: list[dict] = []

    def update_min_mix(best_local: int) -> None:
        nonlocal min_mix
        latent_d = cosine_distance_all(Xn, best_local)
        mismatch = (cat_codes != cat_codes[best_local]).astype(np.float64)
        cat_d = mismatch @ cat_weights
        mix = alpha * latent_d + (1.0 - alpha) * cat_d
        min_mix[:] = np.minimum(min_mix, mix)
        min_mix[best_local] = -np.inf

    for step in range(budget):
        eligible_mask = available.copy()
        for cid in selected_clusters:
            eligible_mask[cluster_labels == cid] = False
        eligible = np.flatnonzero(eligible_mask)
        if eligible.size == 0:
            raise RuntimeError(f"Class {class_id}: no eligible candidates remain")

        diversity_rank = np.ones(eligible.size, dtype=np.float64) if step == 0 else rank_normalize(min_mix[eligible])

        rep = np.empty(eligible.size, dtype=np.float64)
        for j, idx in enumerate(eligible):
            cid = int(cluster_labels[idx])
            info = cluster_info[cid]
            loc = np.where(info["local"] == idx)[0]
            if loc.size != 1:
                raise RuntimeError("Candidate-to-cluster mapping is not unique")
            rep[j] = info["rep_rank"][int(loc[0])]

        score = (
            W_REP * rep
            + W_BOUNDARY * boundary_rank[eligible]
            + W_DIVERSITY * diversity_rank
        )
        order = np.lexsort(
            (
                candidate_positions[eligible],
                -diversity_rank,
                -boundary_rank[eligible],
                -score,
            )
        )
        best = int(eligible[order[0]])
        cid = int(cluster_labels[best])
        loc = np.where(cluster_info[cid]["local"] == best)[0]
        loc_idx = int(loc[0])

        selected.append(best)
        selected_clusters.add(cid)
        available[best] = False

        audit_rows.append(
            {
                "class": class_id,
                "step": step + 1,
                "source_train_position": int(candidate_positions[best]),
                "cluster_id": cid,
                "cluster_size": int(len(cluster_info[cid]["local"])),
                "prototype_distance": float(cluster_info[cid]["dist"][loc_idx]),
                "representativeness_rank": float(cluster_info[cid]["rep_rank"][loc_idx]),
                "teacher_uncertainty": float(uncertainty[best]),
                "same_distance": float(same_mean[best]),
                "opp_distance": float(opp_mean[best]),
                "boundary_support": float(support[best]),
                "boundary_raw": float(boundary_raw[best]),
                "boundary_rank": float(boundary_rank[best]),
                "selection_score": float(score[order[0]]),
            }
        )

        update_min_mix(best)

    selected = np.asarray(selected, dtype=np.int64)
    return candidate_positions[selected], pd.DataFrame(audit_rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--candidate-indices", required=True)
    ap.add_argument("--representation-c0", required=True)
    ap.add_argument("--representation-c1", required=True)
    ap.add_argument("--boundary-representation", required=True)
    ap.add_argument("--teacher-probs", required=True)
    ap.add_argument("--pfi", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--budget-c0", type=int, default=389)
    ap.add_argument("--budget-c1", type=int, default=244)
    ap.add_argument("--alpha", type=float, default=0.8)
    ap.add_argument("--k-same", type=int, default=10)
    ap.add_argument("--k-opp", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-init", type=int, default=10)
    args = ap.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0,1]")

    np.random.seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    train = pd.read_csv(args.train)
    label_col = load_label_col(train)
    labels = train[label_col].to_numpy(dtype=np.int64)

    candidates = np.asarray(np.load(args.candidate_indices), dtype=np.int64)
    rep0 = np.load(args.representation_c0, mmap_mode="r")
    rep1 = np.load(args.representation_c1, mmap_mode="r")
    boundary_rep = np.load(args.boundary_representation, mmap_mode="r")
    probs = np.load(args.teacher_probs, mmap_mode="r")

    N = len(train)
    expected_shape = (N, 73)
    for name, arr in [
        ("representation-c0", rep0),
        ("representation-c1", rep1),
        ("boundary-representation", boundary_rep),
    ]:
        if arr.shape != expected_shape:
            raise ValueError(f"{name}: expected {expected_shape}, got {arr.shape}")
    if probs.shape != (N, 2):
        raise ValueError(f"Expected teacher probs {(N, 2)}, got {probs.shape}")

    if candidates.ndim != 1:
        raise ValueError(f"candidate-indices must be 1-D, got {candidates.shape}")
    if len(np.unique(candidates)) != len(candidates):
        raise ValueError("candidate-indices contains duplicates")
    if candidates.min() < 0 or candidates.max() >= N:
        raise ValueError("candidate index out of range")

    candidate_labels = labels[candidates]
    c0_candidates = candidates[candidate_labels == 0]
    c1_candidates = candidates[candidate_labels == 1]
    if len(c0_candidates) != 1772 or len(c1_candidates) != 760:
        raise ValueError(
            f"Expected candidate counts C0=1772/C1=760, got C0={len(c0_candidates)}/C1={len(c1_candidates)}"
        )

    pfi0, pfi1 = load_pfi_weights(Path(args.pfi))
    cat0 = factorize_categories(train, c0_candidates)
    cat1 = factorize_categories(train, c1_candidates)
    w0 = np.array([pfi0[c] for c in CATEGORICAL_COLS], dtype=np.float64)
    w1 = np.array([pfi1[c] for c in CATEGORICAL_COLS], dtype=np.float64)

    print("=" * 100, flush=True)
    print("BAMD-v3", flush=True)
    print("=" * 100, flush=True)
    print(f"Train rows             : {N}", flush=True)
    print(f"Candidate pool         : {len(candidates)}", flush=True)
    print(f"C0 candidates / budget : {len(c0_candidates)} / {args.budget_c0}", flush=True)
    print(f"C1 candidates / budget : {len(c1_candidates)} / {args.budget_c1}", flush=True)
    print(f"alpha latent           : {args.alpha}", flush=True)
    print(f"PFI C0 categorical     : {pfi0}", flush=True)
    print(f"PFI C1 categorical     : {pfi1}", flush=True)
    print(flush=True)
    print("Selecting Class 0...", flush=True)

    selected_c0, audit0 = select_class(
        class_id=0,
        candidate_positions=c0_candidates,
        weighted_rep=rep0,
        boundary_rep=boundary_rep,
        probs=probs,
        labels=labels,
        cat_codes=cat0,
        cat_weights=w0,
        budget=args.budget_c0,
        alpha=args.alpha,
        k_same=args.k_same,
        k_opp=args.k_opp,
        seed=args.seed,
        n_init=args.n_init,
    )
    print(f"Selected C0: {len(selected_c0)}", flush=True)

    print("Selecting Class 1...", flush=True)
    selected_c1, audit1 = select_class(
        class_id=1,
        candidate_positions=c1_candidates,
        weighted_rep=rep1,
        boundary_rep=boundary_rep,
        probs=probs,
        labels=labels,
        cat_codes=cat1,
        cat_weights=w1,
        budget=args.budget_c1,
        alpha=args.alpha,
        k_same=args.k_same,
        k_opp=args.k_opp,
        seed=args.seed,
        n_init=args.n_init,
    )
    print(f"Selected C1: {len(selected_c1)}", flush=True)

    final_positions = np.concatenate([selected_c0, selected_c1])
    if len(final_positions) != args.budget_c0 + args.budget_c1:
        raise RuntimeError("Final budget mismatch")
    if len(np.unique(final_positions)) != len(final_positions):
        raise RuntimeError("Final subset contains duplicates")

    final_df = train.iloc[final_positions].copy().reset_index(drop=True)
    final_df.insert(0, "source_train_position", final_positions)
    subset_path = out / "model_subset_bamd_v3.csv"
    final_df.to_csv(subset_path, index=False)

    audit = pd.concat([audit0, audit1], ignore_index=True)
    audit.to_csv(out / "bamd_v3_manifest.csv", index=False)
    pd.DataFrame({"source_train_position": selected_c0}).to_csv(out / "bamd_v3_c0_selected_389.csv", index=False)
    pd.DataFrame({"source_train_position": selected_c1}).to_csv(out / "bamd_v3_c1_selected_244.csv", index=False)

    metadata = {
        "method": "BAMD-v3",
        "candidate_pool": len(candidates),
        "candidate_c0": len(c0_candidates),
        "candidate_c1": len(c1_candidates),
        "budget_c0": args.budget_c0,
        "budget_c1": args.budget_c1,
        "total": len(final_positions),
        "weights": {
            "representativeness": W_REP,
            "bounded_boundary": W_BOUNDARY,
            "mixed_diversity": W_DIVERSITY,
        },
        "alpha_latent": args.alpha,
        "boundary_formula": "uncertainty * same_mean/(same_mean+opp_mean+eps)",
        "k_same": args.k_same,
        "k_opp": args.k_opp,
        "categorical_pfi_c0": pfi0,
        "categorical_pfi_c1": pfi1,
        "seed": args.seed,
        "representation_c0": str(args.representation_c0),
        "representation_c1": str(args.representation_c1),
        "boundary_representation": str(args.boundary_representation),
        "teacher_probs": str(args.teacher_probs),
        "pfi": str(args.pfi),
        "subset": str(subset_path),
    }
    (out / "config.json").write_text(json.dumps(metadata, indent=2))

    counts = final_df[label_col].value_counts().sort_index().to_dict()
    print("=" * 100, flush=True)
    print("BAMD-v3 COMPLETE", flush=True)
    print("=" * 100, flush=True)
    print(f"Final class counts     : {counts}", flush=True)
    print(f"Final subset           : {subset_path}", flush=True)
    print(f"Manifest               : {out / 'bamd_v3_manifest.csv'}", flush=True)


if __name__ == "__main__":
    main()
