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
from utils import extract_area_name
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class TestDataset(Dataset):
    def __init__(self, base_folder, feature_columns=None, id_columns=('File', 'Minute', 'Time')):
        self.base_folder = base_folder
        self.id_columns = id_columns
        self.feature_columns = feature_columns or [
            'ClksThisMin', 'medianKHz', 'avSPL', 'avPkAt', 'AvPRF', 'avEndF',
            'tWUTrisk', 'nActualClx', 'nRisingIPIs', 'TrDur_us', 'nICIrising',
            'MinICI_us', 'midpointICI', 'MaxICI_us', 'ClkNofMinICI', 'ClkNofMaxICI',
            'NofClstrs', 'avClstrNx8', 'avPkIPI', 'BeforeIPIratio', 'PreIPIratio',
            'Post1IPIratio', 'Post2IPIratio', 'EndIPIratio'
        ]

        self.labeled_data, self.labels = self._load_all_labeled_data()

    def _load_all_labeled_data(self):
        labeled_data = []
        labels = []

        label_map = {
            "FeedingBuzzes": 0,
            "NonFeedingClickTrains": 1
        }

        for class_folder, label in label_map.items():
            folder_path = os.path.join(self.base_folder, class_folder)
            excel_file = os.path.join(folder_path, f"{class_folder}.xlsx")

            if not os.path.exists(excel_file):
                raise FileNotFoundError(f"Excel file not found: {excel_file}")

            df = pd.read_excel(excel_file)

            df[self.id_columns[0]] = df[self.id_columns[0]].astype(str)
            df[self.id_columns[1]] = df[self.id_columns[1]].astype(int)
            df[self.id_columns[2]] = df[self.id_columns[2]].astype(int)

            df['label'] = label
            labeled_data.append(df)
            labels.extend([label] * len(df))

        full_df = pd.concat(labeled_data, ignore_index=True)

        return full_df, np.array(labels, dtype=np.int64)

    def get_labeled(self):
        df = self.labeled_data[self.feature_columns].apply(
            lambda col: col.astype(str).str.replace(',', '.').astype(np.float32)
        )

        X = df.to_numpy()
        y = self.labels
        meta = self.labeled_data[['File', 'Minute', 'Time']].values
        vectorized_extractor = np.vectorize(extract_area_name)

        clean_unlabeled_meta = vectorized_extractor(meta[:, 0])
        cleaned_meta = np.column_stack((clean_unlabeled_meta, meta[:, 1:3]))
        return X, y, cleaned_meta

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labeled_data.iloc[idx]
        features = row[self.feature_columns].astype(str).str.replace(',', '.').astype(np.float32).to_numpy()
        label = self.labels[idx]
        return features, label

class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]