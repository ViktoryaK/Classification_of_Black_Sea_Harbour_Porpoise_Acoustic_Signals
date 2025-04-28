from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, roc_auc_score, classification_report, confusion_matrix
import re
from collections import defaultdict
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import label_binarize
import os
from tqdm import tqdm
import shap
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
import zipfile
import gdown
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def apply_smote(labeled_x, labeled_y, random_state=42):
    """
    Applies SMOTE oversampling to balance class distribution in labeled data.

    Args:
        labeled_x (np.ndarray): Feature array of shape (n_samples, ...) for labeled data.
        labeled_y (np.ndarray): Corresponding class labels.
        random_state (int): Random seed for reproducibility.

    Returns:
        X_resampled (np.ndarray): Resampled feature array in original shape.
        y_resampled (np.ndarray): Resampled labels.
    """
    print("Original class distribution:", Counter(labeled_y))

    X = labeled_x.reshape((labeled_x.shape[0], -1))

    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, labeled_y)

    print("Resampled class distribution:", Counter(y_resampled))

    original_shape = labeled_x.shape[1:]
    X_resampled = X_resampled.reshape((X_resampled.shape[0],) + original_shape)

    return X_resampled, y_resampled


def find_knn_per_class(
        labeled_x, labeled_y,
        unlabeled_x, unlabeled_meta=None,
        target_class=0,
        n_to_add=20,
        max_K=10,
        distance_threshold=50
):
    """
    For labeled samples of a given class, find neighbors in unlabeled_x
    and select the n_to_add closest ones total for the class.

    Returns:
    - results: list of neighbor info per labeled sample
    - new_labeled_x: np.ndarray of new samples
    - new_labeled_y: np.ndarray of pseudo-labels (same as target_class)
    - updated_unlabeled_x: np.ndarray of unlabeled_x with selected neighbors removed
    - updated_unlabeled_meta: same shape as unlabeled_meta (if given), else None
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    class_mask = labeled_y == target_class
    class_x = labeled_x[class_mask]
    class_indices = np.where(class_mask)[0]

    if len(unlabeled_x) == 0:
        return [], np.empty((0, labeled_x.shape[1])), np.array([]), unlabeled_x, unlabeled_meta

    nn = NearestNeighbors(n_neighbors=min(max_K * 3, len(unlabeled_x)), metric='euclidean')
    nn.fit(unlabeled_x)

    all_candidates = []
    candidate_sources = []

    for l_idx, sample in zip(class_indices, class_x):
        dists, nbrs = nn.kneighbors(sample.reshape(1, -1), return_distance=True)
        dists = dists[0]
        nbrs = nbrs[0]

        for i, d in zip(nbrs, dists):
            if 0 < d <= distance_threshold:
                all_candidates.append((i, d))
                candidate_sources.append((l_idx, sample))

    if not all_candidates:
        return [], np.empty((0, labeled_x.shape[1])), np.array([]), unlabeled_x, unlabeled_meta

    all_candidates_sorted = sorted(all_candidates, key=lambda x: x[1])

    selected_indices = []
    used_unlabeled = set()

    for i, (unlabeled_idx, dist) in enumerate(all_candidates_sorted):
        if unlabeled_idx not in used_unlabeled:
            selected_indices.append(unlabeled_idx)
            used_unlabeled.add(unlabeled_idx)
        if len(selected_indices) >= n_to_add:
            break

    new_labeled_x = unlabeled_x[selected_indices]
    new_labeled_y = np.full(len(new_labeled_x), target_class)

    mask = np.ones(len(unlabeled_x), dtype=bool)
    mask[selected_indices] = False
    updated_unlabeled_x = unlabeled_x[mask]

    if unlabeled_meta is not None:
        updated_unlabeled_meta = unlabeled_meta[mask]
    else:
        updated_unlabeled_meta = None

    results = {
        'selected_indices': selected_indices,
        'num_added': len(selected_indices)
    }

    return results, new_labeled_x, new_labeled_y, updated_unlabeled_x, updated_unlabeled_meta


def extract_area_name(entry):
    special_areas = ['U Dzha Bay', 'U Dzha Sea']
    entry_str = str(entry)
    for area in special_areas:
        if entry_str.startswith(area):
            return area
    match = re.match(r'^(\w+ \w+)', entry_str)
    return match.group(1) if match else entry_str
