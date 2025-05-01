import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap.umap_ as umap  # using the umap-learn package


class TrainDataset(Dataset):
    """
    PyTorch Dataset for training and evaluation.
    """

    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def apply_smote(labeled_x, labeled_y, random_state=42):
    """
    Apply SMOTE to balance class distribution in labeled data.

    Args:
        labeled_x (np.ndarray): Input features of shape (n_samples, ...).
        labeled_y (np.ndarray): Corresponding labels.
        random_state (int): Seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Resampled features and labels.
    """
    print("Original class distribution:", Counter(labeled_y))
    X = labeled_x.reshape((labeled_x.shape[0], -1))
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, labeled_y)
    print("Resampled class distribution:", Counter(y_resampled))

    X_resampled = X_resampled.reshape((X_resampled.shape[0],) + labeled_x.shape[1:])
    return X_resampled, y_resampled


def find_knn_per_class(labeled_x, labeled_y, unlabeled_x, unlabeled_meta=None,
                       target_class=0, n_to_add=20, max_K=10, distance_threshold=50):
    """
    Find nearest neighbors in unlabeled data for a given class and pseudo-label them.

    Args:
        labeled_x (np.ndarray): Features of labeled data.
        labeled_y (np.ndarray): Labels of labeled data.
        unlabeled_x (np.ndarray): Unlabeled data to search for neighbors.
        unlabeled_meta (np.ndarray, optional): Metadata for unlabeled samples.
        target_class (int): Class to find neighbors for.
        n_to_add (int): Number of neighbors to add.
        max_K (int): Number of neighbors to consider per sample.
        distance_threshold (float): Max distance for considering neighbors.

    Returns:
        Tuple: (result dict, new labeled x/y, updated unlabeled_x/meta)
    """
    class_mask = labeled_y == target_class
    class_x = labeled_x[class_mask]

    if len(unlabeled_x) == 0:
        return [], np.empty((0, labeled_x.shape[1])), np.array([]), unlabeled_x, unlabeled_meta

    nn = NearestNeighbors(n_neighbors=min(max_K * 3, len(unlabeled_x)), metric='euclidean')
    nn.fit(unlabeled_x)

    all_candidates = []
    for sample in class_x:
        dists, nbrs = nn.kneighbors(sample.reshape(1, -1), return_distance=True)
        for i, d in zip(nbrs[0], dists[0]):
            if 0 < d <= distance_threshold:
                all_candidates.append((i, d))

    if not all_candidates:
        return [], np.empty((0, labeled_x.shape[1])), np.array([]), unlabeled_x, unlabeled_meta

    all_candidates_sorted = sorted(all_candidates, key=lambda x: x[1])
    selected_indices = []
    used = set()

    for idx, _ in all_candidates_sorted:
        if idx not in used:
            selected_indices.append(idx)
            used.add(idx)
        if len(selected_indices) >= n_to_add:
            break

    new_labeled_x = unlabeled_x[selected_indices]
    new_labeled_y = np.full(len(new_labeled_x), target_class)

    mask = np.ones(len(unlabeled_x), dtype=bool)
    mask[selected_indices] = False
    updated_unlabeled_x = unlabeled_x[mask]
    updated_unlabeled_meta = unlabeled_meta[mask] if unlabeled_meta is not None else None

    results = {'selected_indices': selected_indices, 'num_added': len(selected_indices)}
    return results, new_labeled_x, new_labeled_y, updated_unlabeled_x, updated_unlabeled_meta


def plot_roc_curves(test_y, probs):
    """
    Plot ROC curves for binary classification.

    Args:
        test_y (np.ndarray): True labels.
        probs (np.ndarray): Predicted probabilities.
    """
    y_bin = label_binarize(test_y, classes=[0, 1])
    if y_bin.shape[1] == 1:
        y_bin = np.hstack((1 - y_bin, y_bin))

    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(2):
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green']
    for i, color in zip(range(2), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()


def normalize_and_shuffle(X, y, seed=42):
    """
    Normalize features and shuffle dataset.

    Args:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Labels.
        seed (int): Random seed.

    Returns:
        Tuple[np.ndarray, np.ndarray, StandardScaler]: Shuffled X, y, and scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    indices = np.arange(X.shape[0])
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)

    return X_scaled[indices], y[indices], scaler


def balance_test_set(test_x, test_y, test_meta, seed=42):
    """
    Create a balanced test set with equal samples from each class.

    Args:
        test_x (np.ndarray): Test features.
        test_y (np.ndarray): Test labels.
        test_meta (np.ndarray): Metadata.
        seed (int): Seed for reproducibility.

    Returns:
        Tuple: Balanced test_x, test_y, test_meta
    """
    np.random.seed(seed)
    class0_idx = np.where(test_y == 0)[0]
    class1_idx = np.where(test_y == 1)[0]
    selected_class1_idx = np.random.choice(class1_idx, size=len(class0_idx), replace=False)

    balanced_idx = np.concatenate([class0_idx, selected_class1_idx])
    np.random.shuffle(balanced_idx)

    return test_x[balanced_idx], test_y[balanced_idx], test_meta[balanced_idx]


def print_class_distribution(name, labels):
    """
    Print class distribution in dataset.

    Args:
        name (str): Dataset name.
        labels (np.ndarray): Label array.
    """
    print(f"\n{name} Distribution:")
    for label, count in Counter(labels).items():
        print(f"  Class {label}: {count} ({100 * count / len(labels):.2f}%)")


def prepare_dataloaders(train_x, train_y, val_x, val_y, test_x, test_y, batch_size=64):
    """
    Create PyTorch DataLoaders for training, validation, and testing.

    Args:
        train_x, train_y, val_x, val_y, test_x, test_y: Dataset splits.
        batch_size (int): Batch size.

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]
    """
    train_loader = DataLoader(TrainDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TrainDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TrainDataset(test_x, test_y), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path="training_curves_mlp.pdf"):
    """
    Plot training loss and accuracy over epochs.

    Args:
        train_losses (List[float])
        val_losses (List[float])
        train_accs (List[float])
        val_accs (List[float])
        save_path (str): Path to save the figure.
    """
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Curve')
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()





def plot_pca(labeled_x, unlabeled_x=None, labeled_y=None, title="PCA Visualization", figsize=(8, 6), save_path=None):
    """Plots 2D PCA of labeled and unlabeled data, and optionally saves the figure."""
    if unlabeled_x is not None:
        combined_x = np.vstack((labeled_x, unlabeled_x))
    else:
        combined_x = labeled_x

    pca = PCA(n_components=2)
    reduced = pca.fit_transform(combined_x)

    plt.figure(figsize=figsize)
    if unlabeled_x is not None:
        n_labeled = len(labeled_x)
        unlabeled_reduced = reduced[n_labeled:]
        labeled_reduced = reduced[:n_labeled]
        plt.scatter(unlabeled_reduced[:, 0], unlabeled_reduced[:, 1], c='gray', alpha=0.4, label='Unlabeled', s=20)
        if labeled_y is not None:
            for class_id in np.unique(labeled_y):
                idx = labeled_y == class_id
                plt.scatter(labeled_reduced[idx, 0], labeled_reduced[idx, 1], label=f'Class {class_id}', s=30)
        else:
            plt.scatter(labeled_reduced[:, 0], labeled_reduced[:, 1], label='Labeled', s=30)

    else:
        if labeled_y is not None:
            for class_id in np.unique(labeled_y):
                idx = labeled_y == class_id
                plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f'Class {class_id}', s=30)
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], s=30)

    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Figure saved as PDF to {save_path}")
    else:
        plt.show()


def plot_tsne(labeled_x, unlabeled_x=None, labeled_y=None, title="t-SNE Visualization", figsize=(8, 6), perplexity=30,
              random_state=42, save_path=None):
    """Plots 2D t-SNE of labeled and unlabeled data, and optionally saves the figure."""
    if unlabeled_x is not None:
        combined_x = np.vstack((labeled_x, unlabeled_x))
    else:
        combined_x = labeled_x

    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    reduced = tsne.fit_transform(combined_x)

    plt.figure(figsize=figsize)
    if unlabeled_x is not None:
        n_labeled = len(labeled_x)
        unlabeled_reduced = reduced[n_labeled:]
        labeled_reduced = reduced[:n_labeled]
        plt.scatter(unlabeled_reduced[:, 0], unlabeled_reduced[:, 1], c='gray', alpha=0.4, label='Unlabeled', s=20)
        if labeled_y is not None:
            for class_id in np.unique(labeled_y):
                idx = labeled_y == class_id
                plt.scatter(labeled_reduced[idx, 0], labeled_reduced[idx, 1], label=f'Class {class_id}', s=30)
        else:
            plt.scatter(labeled_reduced[:, 0], labeled_reduced[:, 1], label='Labeled', s=30)

    else:
        if labeled_y is not None:
            for class_id in np.unique(labeled_y):
                idx = labeled_y == class_id
                plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f'Class {class_id}', s=30)
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], s=30)

    plt.title(title)
    plt.xlabel("t-SNE 1")
    plt.ylabel("t-SNE 2")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"Figure saved as PDF to {save_path}")
    else:
        plt.show()


from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
import umap


def plot_umap(labeled_x, unlabeled_x=None, labeled_y=None, title="UMAP Visualization", figsize=(8, 6),
              n_neighbors=15, min_dist=0.1, random_state=42, save_path=None, save_path_outliers=None,
              perform_outlier_detection=False):
    """Plots and optionally saves UMAP projection and outlier detection with Isolation Forest."""

    if unlabeled_x is not None:
        combined_x = np.vstack((labeled_x, unlabeled_x))
    else:
        combined_x = labeled_x

    # Step 1: Compute UMAP projection
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
    reduced = reducer.fit_transform(combined_x)

    # Step 2: Plot normal UMAP
    plt.figure(figsize=figsize)
    if unlabeled_x is not None:
        n_labeled = len(labeled_x)
        unlabeled_reduced = reduced[n_labeled:]
        labeled_reduced = reduced[:n_labeled]
        plt.scatter(unlabeled_reduced[:, 0], unlabeled_reduced[:, 1], c='gray', alpha=0.4, label='Unlabeled', s=20)
        if labeled_y is not None:
            for class_id in np.unique(labeled_y):
                idx = labeled_y == class_id
                plt.scatter(labeled_reduced[idx, 0], labeled_reduced[idx, 1], label=f'Class {class_id}', s=30)
        else:
            plt.scatter(labeled_reduced[:, 0], labeled_reduced[:, 1], label='Labeled', s=30)
    else:
        if labeled_y is not None:
            for class_id in np.unique(labeled_y):
                idx = labeled_y == class_id
                plt.scatter(reduced[idx, 0], reduced[idx, 1], label=f'Class {class_id}', s=30)
        else:
            plt.scatter(reduced[:, 0], reduced[:, 1], s=30)

    plt.title(title)
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', format='pdf')
        print(f"UMAP plot saved to {save_path}")
    else:
        plt.show()

    # Step 3: Outlier detection (if requested)
    if perform_outlier_detection and save_path_outliers:
        iso = IsolationForest(contamination=0.05, random_state=random_state)
        outliers = iso.fit_predict(combined_x)
        is_outlier = (outliers == -1)

        plt.figure(figsize=figsize)
        plt.scatter(reduced[~is_outlier, 0], reduced[~is_outlier, 1], c='blue', label='Inlier', s=5, alpha=0.6)
        plt.scatter(reduced[is_outlier, 0], reduced[is_outlier, 1], c='red', label='Outlier', s=10)
        plt.title("UMAP with Outlier Detection (Isolation Forest)")
        plt.xlabel("UMAP 1")
        plt.ylabel("UMAP 2")
        plt.legend()
        plt.grid(True)

        plt.savefig(save_path_outliers, bbox_inches='tight', format='pdf')
        print(f"UMAP with outliers plot saved to {save_path_outliers}")
