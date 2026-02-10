#!/usr/bin/env python3
"""
Train / evaluate a KNN PHAS classifier from precomputed features.

Input TSV columns (tab-delimited):
    Species
    Identifier
    cID
    type
    complexity
    strand_bias
    log_clust_len_norm_counts
    ratio_abund_len_phasing
    phasis_score

Two modes:

1) Training + 5×(80/20) CV + save model
   python knn_phasis_training.py \
       --mode train \
       --input diamond_set.tsv \
       --model-output knn_model.pkl

2) Evaluation of an existing model (no training)
   python knn_phasis_training.py \
       --mode eval \
       --input diamond_set.tsv \
       --model-input knn_model.pkl
"""

import argparse
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib


FEATURE_COLS = [
    "complexity",
    "strand_bias",
    "log_clust_len_norm_counts",
    "ratio_abund_len_phasing",
    "phasis_score",
]


def load_feature_table(path: str) -> pd.DataFrame:
    """Load the tab-delimited feature table."""
    df = pd.read_csv(path, sep="\t")
    missing = [c for c in FEATURE_COLS + ["type"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) in input file: {missing}")
    return df


def extract_X_y(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract feature matrix X and label vector y from the dataframe.

    Returns:
        X: 2D numpy array of shape (n_samples, n_features)
        y: 1D numpy array of shape (n_samples,) with string labels
        idx: 1D numpy array of indices kept (useful if rows are dropped)
    """
    # Keep track of original index for transparency
    idx = df.index.to_numpy()

    # Extract labels
    y = df["type"].astype(str).str.strip().to_numpy()

    # Extract features and coerce to float
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").to_numpy()

    # Drop rows with any NaNs in features or labels
    valid_mask = ~np.isnan(X).any(axis=1)
    if y.dtype.kind == "O":
        # y is already string; NaN in labels is unlikely, but just in case:
        valid_mask &= pd.notna(df["type"]).to_numpy()

    dropped = len(df) - valid_mask.sum()
    if dropped > 0:
        print(f"[info] Dropped {dropped} row(s) with NaNs in features or labels.")

    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    idx_valid = idx[valid_mask]

    return X_valid, y_valid, idx_valid


def run_repeated_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 5,
    test_size: float = 0.2,
    n_neighbors: int = 5,
    base_seed: int = 0,
) -> KNeighborsClassifier:
    """
    Perform repeated 80/20 CV (5 times by default) on scaled features.

    Scaling is done once on the full dataset (MinMaxScaler.fit_transform),
    matching the original notebook behaviour.

    Returns:
        knn_clf from the *last* repetition (this is what gets saved as knn_model.pkl).
    """
    print("=== Repeated 80/20 CV ===")
    print(
        f"  n_samples = {X.shape[0]}, n_features = {X.shape[1]}, "
        f"n_repeats = {n_repeats}, test_size = {test_size}, "
        f"n_neighbors = {n_neighbors}"
    )

    # Scale once on the full dataset (as in the original notebook)
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    classes = np.unique(y)
    print(f"  Classes: {classes.tolist()}")

    metrics_per_run: List[dict] = []
    cm_sum = None

    knn_clf = None  # will be set each run

    for i in range(n_repeats):
        rs = base_seed + i
        print(f"\n--- Run {i + 1}/{n_repeats} (random_state={rs}) ---")

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled,
            y,
            test_size=test_size,
            random_state=rs,
            shuffle=True,
            stratify=y,
        )

        knn_clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights="distance",
            algorithm="auto",
            p=2,
            n_jobs=-1,
        )

        knn_clf.fit(X_train, y_train)
        y_pred = knn_clf.predict(X_test)

        # Confusion matrix for this run
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        cm_df = pd.DataFrame(
            cm,
            index=[f"true_{c}" for c in classes],
            columns=[f"pred_{c}" for c in classes],
        )
        print("\nConfusion matrix (this run):")
        print(cm_df)

        # Classification report for this run
        print("\nClassification report (this run):")
        print(classification_report(y_test, y_pred, digits=3))

        # Aggregate confusion matrix
        if cm_sum is None:
            cm_sum = cm
        else:
            cm_sum = cm_sum + cm

        # Macro-averaged metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        metrics_per_run.append(
            {
                "run": i + 1,
                "accuracy": acc,
                "precision_macro": prec,
                "recall_macro": rec,
                "f1_macro": f1,
            }
        )

        print(
            f"\nSummary (this run): "
            f"accuracy={acc:.3f}, precision_macro={prec:.3f}, "
            f"recall_macro={rec:.3f}, f1_macro={f1:.3f}"
        )

    # Aggregate metrics across runs
    metrics_df = pd.DataFrame(metrics_per_run)
    print("\n=== Metrics across runs ===")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n=== Mean ± SD (macro-averaged metrics) ===")
    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        mean = metrics_df[metric].mean()
        std = metrics_df[metric].std(ddof=1)
        print(f"{metric:16s}: {mean:.3f} ± {std:.3f}")

    # Aggregate confusion matrix across runs
    if cm_sum is not None:
        print("\n=== Confusion matrix summed over all runs ===")
        cm_sum_df = pd.DataFrame(
            cm_sum,
            index=[f"true_{c}" for c in classes],
            columns=[f"pred_{c}" for c in classes],
        )
        print(cm_sum_df)

    # Return the last trained classifier (as the notebook did)
    return knn_clf


def evaluate_existing_model(
    model_path: str,
    X: np.ndarray,
    y: np.ndarray,
) -> None:
    """
    Evaluate an existing knn_model.pkl on a feature table.

    Behaviour mimics phasis-classifier usage:
    - Scale the full dataset with MinMaxScaler.fit_transform,
      then call model.predict on the scaled features.
    """
    print(f"[info] Loading model from: {model_path}")
    knn_clf = joblib.load(model_path)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    y_pred = knn_clf.predict(X_scaled)
    classes = np.unique(y)

    cm = confusion_matrix(y, y_pred, labels=classes)
    cm_df = pd.DataFrame(
        cm,
        index=[f"true_{c}" for c in classes],
        columns=[f"pred_{c}" for c in classes],
    )

    print("\n=== Evaluation of existing model on full dataset ===")
    print("\nConfusion matrix:")
    print(cm_df)

    print("\nClassification report:")
    print(classification_report(y, y_pred, digits=3))

    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average="macro", zero_division=0)
    rec = recall_score(y, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y, y_pred, average="macro", zero_division=0)

    print(
        f"\nSummary (full dataset): "
        f"accuracy={acc:.3f}, precision_macro={prec:.3f}, "
        f"recall_macro={rec:.3f}, f1_macro={f1:.3f}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate KNN PHAS classifier from precomputed features."
    )
    parser.add_argument(
        "--mode",
        choices=["train", "eval"],
        default="train",
        help="Mode: 'train' = train + 5×(80/20) CV + save model; "
             "'eval' = evaluate existing model on full dataset.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Tab-delimited input file with features and 'type' column.",
    )
    parser.add_argument(
        "--model-output",
        default="knn_model.pkl",
        help="Output path for trained KNN model (train mode).",
    )
    parser.add_argument(
        "--model-input",
        help="Path to an existing knn_model.pkl (required for eval mode).",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=5,
        help="Number of repeated 80/20 splits for CV (train mode).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test size fraction for each split (train mode).",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Number of neighbors (k) in KNN.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed; each run uses seed + run_index.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    df = load_feature_table(args.input)
    X, y, _ = extract_X_y(df)

    if X.shape[0] < 10:
        print(
            f"[warning] Only {X.shape[0]} valid rows after filtering; "
            f"metrics may be unstable."
        )

    if args.mode == "train":
        knn_clf = run_repeated_cv(
            X=X,
            y=y,
            n_repeats=args.n_repeats,
            test_size=args.test_size,
            n_neighbors=args.neighbors,
            base_seed=args.seed,
        )

        if knn_clf is None:
            raise RuntimeError("Training failed; no classifier created.")

        print(f"\n[saving] Writing trained KNN model to: {args.model_output}")
        joblib.dump(knn_clf, args.model_output)

    elif args.mode == "eval":
        if not args.model_input:
            raise SystemExit(
                "Error: --model-input is required when --mode eval is used."
            )
        evaluate_existing_model(args.model_input, X, y)


if __name__ == "__main__":
    main()
