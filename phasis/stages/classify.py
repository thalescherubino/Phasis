"""
Classification stage extracted from legacy.py (migration-safe).

IMPORTANT:
- Pure stage: does NOT write files and does NOT plot.
- Keeps legacy semantics: same scaling, same model path, same post-filters.
- No nested functions; no imports inside functions.
"""

import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
import joblib
import warnings

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but KNeighborsClassifier was fitted with feature names",
    category=UserWarning,
)

# Keep the exact feature set that legacy.py uses for KNN/GMM
KNN_FEATURE_COLS = [
    "complexity",
    "strand_bias",
    "log_clust_len_norm_counts",
    "ratio_abund_len_phase",
    "phasis_score",
]


def _default_knn_model_path() -> str:
    """
    legacy.py uses:
        script_dir = os.path.dirname(os.path.realpath(__file__))  # legacy.py lives in phasis/
        model_path = os.path.join(script_dir, "data", "knn_model.pkl")

    This file lives in phasis/stages/, so we go one directory up to phasis/.
    """
    phasis_pkg_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    return os.path.join(phasis_pkg_dir, "data", "knn_model.pkl")


def _apply_post_filters(
    df: pd.DataFrame,
    phasisScoreCutoff: float,
    min_Howell_score: float,
    max_complexity: float,
) -> pd.DataFrame:
    """
    Apply the exact same post-filters as legacy.KNN_phas_clustering/GMM_phas_clustering.
    """
    out = df.copy()

    # 1) Score + Howell filter
    out.loc[
        (out["phasis_score"] < phasisScoreCutoff)
        | (out["Peak_Howell_score"] < min_Howell_score),
        "label",
    ] = "non-PHAS"

    # 2) Complexity filter
    out.loc[(out["complexity"] > max_complexity), "label"] = "non-PHAS"

    return out


def knn_classify(
    features: pd.DataFrame,
    phasisScoreCutoff: float,
    min_Howell_score: float,
    max_complexity: float,
    model_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    KNN classifier (pure): returns `features` with `label` column set.
    """
    # Scale the same KNN feature set as legacy
    X = features[KNN_FEATURE_COLS].copy()
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Load model and predict
    if model_path is None:
        model_path = _default_knn_model_path()
    knn_clf = joblib.load(model_path)
    y_pred = knn_clf.predict(X_scaled)

    out = features.copy()
    out["label"] = np.where(y_pred == 1, "PHAS", "non-PHAS")

    # Post-filters (identical semantics)
    out = _apply_post_filters(out, phasisScoreCutoff, min_Howell_score, max_complexity)
    return out


def gmm_classify(
    features: pd.DataFrame,
    phasisScoreCutoff: float,
    min_Howell_score: float,
    max_complexity: float,
    n_clusters: int = 2,
) -> pd.DataFrame:
    """
    GMM clustering aligned with KNN feature scaling + post-filters.
    Chooses the GMM cluster with the highest mean phasis_score as PHAS.
    """
    cols_for_model = list(KNN_FEATURE_COLS)
    X = features[cols_for_model].copy()
    scaler = preprocessing.MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    cluster_labels = gmm.fit_predict(X_scaled)

    tmp = pd.DataFrame(
        {"cluster": cluster_labels, "phasis_score": features["phasis_score"].values}
    )
    phas_cluster = tmp.groupby("cluster")["phasis_score"].mean().idxmax()

    out = features.copy()
    out["label"] = np.where(cluster_labels == phas_cluster, "PHAS", "non-PHAS")

    out = _apply_post_filters(out, phasisScoreCutoff, min_Howell_score, max_complexity)
    return out