import pandas as pd
import numpy as np
from pathlib import Path
from fastdtw import fastdtw
import random
from scipy.spatial.distance import euclidean
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap
from imblearn.over_sampling import SMOTE
from collections import Counter



class DataProcessor:
    dataset_start_time = pd.Timestamp("1899-12-30")

    @staticmethod
    def read_data(folder: str) -> pd.DataFrame:
        """Reads all files in a folder."""
        files = Path(folder).glob("*")
        return pd.concat((pd.read_csv(file, sep="\t") for file in files), ignore_index=True)

    @staticmethod
    def add_datetime_column(df: pd.DataFrame, time_col: str, micro_col: str) -> pd.DataFrame:
        """Vectorized datetime computation."""
        df["Datetime"] = DataProcessor.dataset_start_time + pd.to_timedelta(df[time_col], unit="m") + pd.to_timedelta(
            df[micro_col], unit="us")
        return df

    @staticmethod
    def process_train_data(df: pd.DataFrame) -> pd.DataFrame:
        df = DataProcessor.add_datetime_column(df, "Minute", "Time")
        df["File"] = df["File"].str.split().str[:2].str.join(" ")
        return df.loc[:, ["File", "Datetime", "medianKHz", "avSPL", "AvPRF"]]

    @staticmethod
    def process_click_data(df: pd.DataFrame) -> pd.DataFrame:
        df = DataProcessor.add_datetime_column(df, "Minute", "microsec")
        df = df.assign(Clk_s=(1e6 / df["ICI"]))
        df["File"] = df["File"].str.split().str[:2].str.join(" ")
        return df.loc[:, ["File", "Datetime", "ClkKHZ", "maxPk", "Clk_s"]]


def dtw_distance(seq1, seq2):
    if len(seq1) < 2 or len(seq2) < 2:
        raise ValueError("Sequences must have at least two points")

    distance, _ = fastdtw(seq1.reshape(-1, 1), seq2.reshape(-1, 1), dist=euclidean)
    return distance


DTW_THRESHOLD = 8


def assign_label(seq1, seq2):
    """Assigns label 0 if sequences are similar, else label 1."""
    distance = dtw_distance(seq1, seq2)
    return 0 if distance < DTW_THRESHOLD else 1


def pad_sequences(seq_list, max_len=None):
    """Pads a list of sequences to the same length using NumPy."""
    if max_len is None:
        max_len = max(len(seq) for seq in seq_list)

    padded_seqs = np.zeros((len(seq_list), max_len, 1))

    for i, seq in enumerate(seq_list):
        padded_seqs[i, :len(seq), 0] = seq.squeeze()
    print("Padded the sequences to len: " + str(max_len))
    return padded_seqs


def create_dtw_pairs(dataset, num_pairs=1000):
    """Generates training pairs with DTW-based similarity labels."""
    pairs_1, pairs_2, labels = [], [], []
    n = len(dataset)

    for _ in range(num_pairs):
        idx1, idx2 = random.sample(range(n), 2)
        seq1, seq2 = dataset[idx1]["Clk/s"].values.reshape(-1, 1), dataset[idx2]["Clk/s"].values.reshape(-1, 1)

        label = assign_label(seq1, seq2)

        pairs_1.append(seq1)
        pairs_2.append(seq2)
        labels.append(label)
    print("Created pairs")
    print("Class 1: ", sum(labels), " Class 0: ", len(labels) - sum(labels))
    pairs_1 = pad_sequences(pairs_1)
    pairs_2 = pad_sequences(pairs_2)
    labels = np.array(labels, dtype=np.float32)

    return pairs_1, pairs_2, labels



def _plot_2d(reduced, labeled_x, labeled_y=None, unlabeled_len=0, pseudo_len=0, pseudo_y=None, title="", method_name="",
             figsize=(8, 6)):
    n_labeled = len(labeled_x)
    start_pseudo = n_labeled
    start_unlabeled = n_labeled + pseudo_len

    plt.figure(figsize=figsize)

    # Unlabeled
    if unlabeled_len > 0:
        unlabeled_reduced = reduced[start_unlabeled:]
        plt.scatter(unlabeled_reduced[:, 0], unlabeled_reduced[:, 1], c='gray', alpha=0.4, label='Unlabeled', s=20)

    # Labeled
    labeled_reduced = reduced[:n_labeled]
    if labeled_y is not None:
        for class_id in np.unique(labeled_y):
            idx = labeled_y == class_id
            plt.scatter(labeled_reduced[idx, 0], labeled_reduced[idx, 1], label=f'Class {class_id}', s=30)
    else:
        plt.scatter(labeled_reduced[:, 0], labeled_reduced[:, 1], label='Labeled', s=30)

    # Pseudo-labeled
    if pseudo_len > 0 and pseudo_y is not None:
        pseudo_reduced = reduced[start_pseudo:start_unlabeled]
        for class_id in np.unique(pseudo_y):
            idx = pseudo_y == class_id
            plt.scatter(pseudo_reduced[idx, 0], pseudo_reduced[idx, 1], alpha=0.9, label=f'Pseudo Class {class_id}', marker='x',
                        s=30)

    plt.title(title or f"{method_name} Visualization")
    plt.xlabel(f"{method_name} 1")
    plt.ylabel(f"{method_name} 2")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_pca(labeled_x, unlabeled_x=None, labeled_y=None, pseudo_x=None, pseudo_y=None, title="PCA Visualization",
             figsize=(8, 6)):
    data_blocks = [labeled_x]
    if pseudo_x is not None:
        data_blocks.append(pseudo_x)
    if unlabeled_x is not None:
        data_blocks.append(unlabeled_x)

    combined_x = np.vstack(data_blocks)
    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined_x)

    _plot_2d(
        reduced, labeled_x, labeled_y,
        unlabeled_len=0 if unlabeled_x is None else len(unlabeled_x),
        pseudo_len=0 if pseudo_x is None else len(pseudo_x),
        pseudo_y=pseudo_y, title=title, method_name="PCA", figsize=figsize
    )


def plot_tsne(labeled_x, unlabeled_x=None, labeled_y=None, pseudo_x=None, pseudo_y=None, title="t-SNE Visualization",
              figsize=(8, 6), perplexity=30, random_state=42):
    data_blocks = [labeled_x]
    if pseudo_x is not None:
        data_blocks.append(pseudo_x)
    if unlabeled_x is not None:
        data_blocks.append(unlabeled_x)

    combined_x = np.vstack(data_blocks)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    reduced = tsne.fit_transform(combined_x)

    _plot_2d(
        reduced, labeled_x, labeled_y,
        unlabeled_len=0 if unlabeled_x is None else len(unlabeled_x),
        pseudo_len=0 if pseudo_x is None else len(pseudo_x),
        pseudo_y=pseudo_y, title=title, method_name="t-SNE", figsize=figsize
    )


def plot_umap(labeled_x, unlabeled_x=None, labeled_y=None, pseudo_x=None, pseudo_y=None, title="UMAP Visualization",
              figsize=(8, 6), n_neighbors=15, min_dist=0.1, random_state=42):
    data_blocks = [labeled_x]
    if pseudo_x is not None:
        data_blocks.append(pseudo_x)
    if unlabeled_x is not None:
        data_blocks.append(unlabeled_x)

    combined_x = np.vstack(data_blocks)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    reduced = reducer.fit_transform(combined_x)

    _plot_2d(
        reduced, labeled_x, labeled_y,
        unlabeled_len=0 if unlabeled_x is None else len(unlabeled_x),
        pseudo_len=0 if pseudo_x is None else len(pseudo_x),
        pseudo_y=pseudo_y, title=title, method_name="UMAP", figsize=figsize
    )


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
    max_K=10,
    distance_threshold=50
):
    """
    For labeled samples of a given class, find up to max_K neighbors in unlabeled_x
    that are within distance_threshold. Avoids reuse and removes neighbors from unlabeled pool.

    Returns:
    - results: list of neighbor info per labeled sample
    - new_labeled_x: np.ndarray of new samples
    - new_labeled_y: np.ndarray of pseudo-labels (same as target_class)
    - updated_unlabeled_x: np.ndarray of unlabeled_x with selected neighbors removed
    - updated_unlabeled_meta: same shape as unlabeled_meta (if given), else None
    """
    class_mask = labeled_y == target_class
    class_x = labeled_x[class_mask]
    class_indices = np.where(class_mask)[0]

    if len(unlabeled_x) == 0:
        return [], np.empty((0, labeled_x.shape[1])), np.array([]), unlabeled_x, unlabeled_meta

    nn = NearestNeighbors(n_neighbors=min(max_K * 3, len(unlabeled_x)), metric='euclidean')
    nn.fit(unlabeled_x)

    results = []
    neighbor_data = []
    neighbor_labels = []
    used_indices = set()

    for l_idx, sample in zip(class_indices, class_x):
        dists, nbrs = nn.kneighbors(sample.reshape(1, -1), return_distance=True)
        dists = dists[0]
        nbrs = nbrs[0]

        selected = [(i, d) for i, d in zip(nbrs, dists)
                    if 0 < d <= distance_threshold and i not in used_indices]

        if selected:
            selected = selected[:max_K]
            sel_indices, sel_dists = zip(*selected)
            used_indices.update(sel_indices)

            neighbor_data.append(unlabeled_x[list(sel_indices)])
            neighbor_labels.append(np.full(len(sel_indices), target_class))

            results.append({
                'labeled_index': l_idx,
                'labeled_sample': sample,
                'neighbor_indices': sel_indices,
                'distances': sel_dists
            })

    if neighbor_data:
        new_labeled_x = np.vstack(neighbor_data)
        new_labeled_y = np.concatenate(neighbor_labels)
    else:
        new_labeled_x = np.empty((0, labeled_x.shape[1]))
        new_labeled_y = np.array([])

    used_indices = sorted(used_indices)
    mask = np.ones(len(unlabeled_x), dtype=bool)
    mask[used_indices] = False
    updated_unlabeled_x = unlabeled_x[mask]

    if unlabeled_meta is not None:
        updated_unlabeled_meta = unlabeled_meta[mask]
    else:
        updated_unlabeled_meta = None

    return results, new_labeled_x, new_labeled_y, updated_unlabeled_x, updated_unlabeled_meta
