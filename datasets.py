import os
import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset


class TestDataset(Dataset):
    """
    PyTorch Dataset for loading labeled test data from Excel files.
    Assumes labeled folders 'FeedingBuzzes' and 'NonFeedingClickTrains',
    each containing a corresponding Excel file with metadata and features.
    """

    def __init__(self, base_folder, feature_columns=None, id_columns=('File', 'Minute', 'Time')):
        """
        Args:
            base_folder (str): Path to the root folder containing labeled Excel files.
            feature_columns (list): List of feature column names. Uses default if None.
            id_columns (tuple): Columns used to identify each instance (File, Minute, Time).
        """
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

    @staticmethod
    def extract_area_name(entry):
        """
        Extracts and standardizes area names from the 'File' column.
        """
        entry_str = str(entry)
        for area in AcousticDataset.special_areas:
            if entry_str.startswith(area):
                return area
        match = re.match(r'^(\w+ \w+)', entry_str)
        return match.group(1) if match else entry_str

    def _load_all_labeled_data(self):
        """
        Loads labeled data from class subfolders and assigns labels.
        """
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

            # Ensure consistent data types
            df[self.id_columns[0]] = df[self.id_columns[0]].astype(str)
            df[self.id_columns[1]] = df[self.id_columns[1]].astype(int)
            df[self.id_columns[2]] = df[self.id_columns[2]].astype(int)

            df['label'] = label
            labeled_data.append(df)
            labels.extend([label] * len(df))

        full_df = pd.concat(labeled_data, ignore_index=True)
        return full_df, np.array(labels, dtype=np.int64)

    def get_labeled(self):
        """
        Returns:
            X (ndarray): Feature matrix.
            y (ndarray): Label vector.
            meta (ndarray): Metadata (File, Minute, Time), with cleaned File names.
        """
        df = self.labeled_data[self.feature_columns].apply(
            lambda col: col.astype(str).str.replace(',', '.').astype(np.float32)
        )
        X = df.to_numpy()
        y = self.labels
        meta = self.labeled_data[list(self.id_columns)].values

        # Clean file names
        cleaned_files = np.vectorize(self.extract_area_name)(meta[:, 0])
        cleaned_meta = np.column_stack((cleaned_files, meta[:, 1:3]))
        return X, y, cleaned_meta

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labeled_data.iloc[idx]
        features = row[self.feature_columns].astype(str).str.replace(',', '.').astype(np.float32).to_numpy()
        label = self.labels[idx]
        return features, label


class AcousticDataset(Dataset):
    """
    Dataset class for semi-supervised acoustic data.
    Loads full dataset from CSV and optionally adds labeled data from an Excel file.
    Automatically removes labeled instances from the unlabeled pool.
    """

    special_areas = ['U Dzha Bay', 'U Dzha Sea']  # Used for area name extraction

    def __init__(self, csv_path, excel_path=None, feature_columns=None, id_columns=('File', 'Minute', 'Time')):
        """
        Args:
            csv_path (str): Path to CSV file containing the main dataset.
            excel_path (str): Optional path to Excel file with labeled data in sheets.
            feature_columns (list): List of feature columns to use.
            id_columns (tuple): Identifier columns for matching (File, Minute, Time).
        """
        self.data = pd.read_csv(csv_path, sep='\t')
        self.id_columns = id_columns
        self.feature_columns = feature_columns or [
            'ClksThisMin', 'medianKHz', 'avSPL', 'avPkAt', 'AvPRF', 'avEndF',
            'tWUTrisk', 'nActualClx', 'nRisingIPIs', 'TrDur_us', 'nICIrising',
            'MinICI_us', 'midpointICI', 'MaxICI_us', 'ClkNofMinICI', 'ClkNofMaxICI',
            'NofClstrs', 'avClstrNx8', 'avPkIPI', 'BeforeIPIratio', 'PreIPIratio',
            'Post1IPIratio', 'Post2IPIratio', 'EndIPIratio'
        ]

        self.label_encoder = LabelEncoder()
        self.labeled_data = pd.DataFrame()
        self.labels = []
        self.labeled_indices = set()

        if excel_path:
            self._load_labeled_data(excel_path)

            # Clean metadata for matching
            self.data[self.id_columns[0]] = np.vectorize(self.extract_area_name)(
                self.data[self.id_columns[0]].astype(str)
            )
            self.data[self.id_columns[1]] = self.data[self.id_columns[1]].astype(int)
            self.data[self.id_columns[2]] = self.data[self.id_columns[2]].astype(int)

            # Exclude labeled samples from the unlabeled dataset
            is_unlabeled = ~self.data[list(self.id_columns)].apply(tuple, axis=1).isin(self.labeled_indices)
            self.unlabeled_data = self.data[is_unlabeled].reset_index(drop=True)

    def _load_labeled_data(self, excel_path):
        """
        Loads labeled data from an Excel file with multiple sheets.
        Uses sheet names as labels.
        """
        xl = pd.ExcelFile(excel_path)
        dfs = []

        for sheet in xl.sheet_names:
            df = xl.parse(sheet)
            df['label'] = sheet
            dfs.append(df)

        full_labeled = pd.concat(dfs, ignore_index=True)

        # Clean for matching with main dataset
        full_labeled[self.id_columns[0]] = np.vectorize(self.extract_area_name)(
            full_labeled[self.id_columns[0]].astype(str)
        )
        full_labeled[self.id_columns[1]] = full_labeled[self.id_columns[1]].astype(int)
        full_labeled[self.id_columns[2]] = full_labeled[self.id_columns[2]].astype(int)

        self.labeled_data = full_labeled
        self.labels = self.label_encoder.fit_transform(full_labeled['label'])
        self.labeled_indices = set(full_labeled[list(self.id_columns)].apply(tuple, axis=1))

    @staticmethod
    def extract_area_name(entry):
        """
        Standardizes area names for consistency across datasets.
        """
        entry_str = str(entry)
        for area in AcousticDataset.special_areas:
            if entry_str.startswith(area):
                return area
        match = re.match(r'^(\w+ \w+)', entry_str)
        return match.group(1) if match else entry_str

    def _clean_metadata(self, meta_df):
        """
        Cleans metadata for consistent use.
        """
        file_names = meta_df[self.id_columns[0]].to_numpy()
        cleaned_files = np.vectorize(self.extract_area_name)(file_names)
        return np.column_stack((cleaned_files, meta_df['Minute'].to_numpy(), meta_df['Time']))

    def get_labeled(self):
        """
        Returns:
            X (ndarray): Feature matrix for labeled data.
            y (ndarray): Binary label vector.
            meta (ndarray): Cleaned metadata (File, Minute, Time).
        """
        df = self.labeled_data[self.feature_columns].apply(
            lambda col: col.astype(str).str.replace(',', '.').astype(np.float32)
        )
        X = df.to_numpy()
        y = np.array(self.labels, dtype=np.int64)

        # Optional relabeling for binary classification
        np.putmask(y, y == 1, 0)
        np.putmask(y, np.isin(y, [2, 3]), 1)

        return X, y, self._clean_metadata(self.labeled_data[list(self.id_columns)])

    def get_unlabeled(self):
        """
        Returns:
            X (ndarray): Feature matrix for unlabeled data.
            meta (ndarray): Cleaned metadata (File, Minute, Time).
        """
        X = self.unlabeled_data[self.feature_columns].to_numpy(dtype=np.float32)
        return X, self._clean_metadata(self.unlabeled_data[list(self.id_columns)])

    def __len__(self):
        return len(self.unlabeled_data)

    def __getitem__(self, idx):
        """
        Returns:
            features (ndarray): Feature vector of a single unlabeled instance.
        """
        features = self.unlabeled_data.iloc[idx][self.feature_columns].to_numpy(dtype=np.float32)
        return features
