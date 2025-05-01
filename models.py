import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# ----------------------------
# Tree-based models (sklearn/xgboost)
# ----------------------------

# Random Forest model with specified hyperparameters
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight='balanced',
    bootstrap=True,
    random_state=42
)

# XGBoost model with specified hyperparameters
xgb_model = XGBClassifier(
    n_estimators=100,
    max_depth=10,
    learning_rate=0.005,
    subsample=0.9,
    colsample_bytree=0.8,
    eval_metric='logloss',
)

# ----------------------------
# Neural network models (PyTorch)
# ----------------------------

class MLPClassifier(nn.Module):
    """
    Multi-layer Perceptron classifier.

    Args:
        input_dim (int): Number of input features.
        hidden_dims (list[int]): Sizes of hidden layers, e.g., [128, 64].
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout probability.
    """
    def __init__(self, input_dim, hidden_dims, num_classes=2, dropout_rate=0.2):
        super(MLPClassifier, self).__init__()
        self.model = self._build_model(input_dim, hidden_dims, num_classes, dropout_rate)

    def _build_model(self, input_dim, hidden_dims, num_classes, dropout_rate):
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.Linear(dims[-1], num_classes))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class AutoencoderClassifier(nn.Module):
    """
    Autoencoder-based classifier that reconstructs input and predicts class label.

    Args:
        input_dim (int): Number of input features.
        hidden_dim (int): Dimension of hidden layers.
        bottleneck_dim (int): Size of bottleneck layer.
        num_classes (int): Number of output classes.
        dropout_rate (float): Dropout probability.
    """
    def __init__(self, input_dim, hidden_dim=128, bottleneck_dim=64, num_classes=2, dropout_rate=0.2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        logits = self.classifier(encoded)
        return reconstructed, logits

# ----------------------------
# Inference functions for MLP and AE models
# ----------------------------

def predict_ae(model, x, device='cuda'):
    """
    Predict class labels using an autoencoder-based classifier.

    Args:
        model (nn.Module): Trained model (returns reconstruction, logits).
        x (np.ndarray or torch.Tensor): Input data.
        device (str): Device to use for computation.

    Returns:
        np.ndarray: Predicted class labels.
    """
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    x = x.to(device)
    with torch.no_grad():
        _, logits = model(x)
        preds = logits.argmax(dim=1)
    return preds.cpu().numpy()


def predict_proba_ae(model, x, device='cuda'):
    """
    Predict class probabilities using an autoencoder-based classifier.

    Args:
        model (nn.Module): Trained model (returns reconstruction, logits).
        x (np.ndarray or torch.Tensor): Input data.
        device (str): Device to use for computation.

    Returns:
        np.ndarray: Predicted probabilities.
    """
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    x = x.to(device)
    with torch.no_grad():
        _, logits = model(x)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()


def predict_mlp(model, x, device='cuda'):
    """
    Predict class labels using a standard MLP classifier.

    Args:
        model (nn.Module): Trained model.
        x (np.ndarray or torch.Tensor): Input data.
        device (str): Device to use for computation.

    Returns:
        np.ndarray: Predicted class labels.
    """
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)
    return preds.cpu().numpy()


def predict_proba_mlp(model, x, device='cuda'):
    """
    Predict class probabilities using a standard MLP classifier.

    Args:
        model (nn.Module): Trained model.
        x (np.ndarray or torch.Tensor): Input data.
        device (str): Device to use for computation.

    Returns:
        np.ndarray: Predicted probabilities.
    """
    model.eval()
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).float()
    x = x.to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
    return probs.cpu().numpy()
