import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
from datetime import datetime
from utils import DataProcessor
from collections import defaultdict


class ClickTrainDataset(Dataset):
    """PyTorch Dataset that extracts polynomial fit coefficients up to the third degree."""

    def __init__(self, train_folder: str, transform=None):
        self.train_data = DataProcessor.process_train_data(DataProcessor.read_data(train_folder))
        self.train_data = self.train_data.sort_values(by=["File", "Datetime"]).reset_index(drop=True)
        self.features = self.extract_features()
        if transform:
            self.features = transform.fit_transform(self.features)
            self.features = pd.DataFrame(self.features)

    def extract_features(self) -> pd.DataFrame:
        """Extracts features efficiently without repeating train rows in memory."""
        return self.train_data[["medianKHz", "avSPL", "AvPRF"]].reset_index(drop=True)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns a tensor of features for a given index."""
        row = self.features.iloc[idx]
        return torch.tensor(row.values, dtype=torch.float32)


class PolynomialDataset(Dataset):
    """PyTorch Dataset that extracts polynomial fit coefficients up to the third degree."""

    def __init__(self, click_folder: str, train_folder: str, transform=None):
        self.train_data = DataProcessor.process_train_data(DataProcessor.read_data(train_folder))
        self.click_data = DataProcessor.process_click_data(DataProcessor.read_data(click_folder))

        self.train_data = self.train_data.sort_values(by=["File", "Datetime"]).reset_index(drop=True)

        self.click_data = self.click_data.sort_values(by=["File", "Datetime"]).reset_index(drop=True)
        self.all_data = self.match_data()
        self.features = self.extract_features()
        if transform:
            self.features = transform.fit_transform(self.features)
            self.features = pd.DataFrame(self.features)

    def match_data(self) -> dict:
        """Associates click data with corresponding train data timestamps efficiently."""
        matched_data = defaultdict(list)

        for file, train_subset in self.train_data.groupby("File"):
            click_subset = self.click_data[self.click_data["File"] == file]
            if click_subset.empty:
                continue

            for i in range(len(train_subset) - 1):
                dt, next_dt = train_subset.iloc[i]["Datetime"], train_subset.iloc[i + 1]["Datetime"]

                clicks_in_range = click_subset[
                    (click_subset["Datetime"] >= dt) & (click_subset["Datetime"] < next_dt)
                    ]
                matched_data[(file, dt)] = [train_subset.iloc[i], clicks_in_range]

        return matched_data

    def extract_features(self) -> pd.DataFrame:
        """Extracts features efficiently by fitting a polynomial to 'Clk/s' over time and returning coefficients."""
        features = []

        for (_, _), (train, click) in self.all_data.items():
            train_values = train.iloc[:, 2:].to_numpy().flatten()
            print(train_values)
            click = click[click["Clk/s"] != np.inf]

            if click.empty:
                continue

            t = (click.index - click.index[0]).total_seconds()
            y = click["Clk/s"].to_numpy()

            coeffs = np.polyfit(t, y, 3) if len(t) > 3 else np.zeros(4)

            clk_ratio = click["Clk/s"].max() / click["Clk/s"].iloc[-1]

            features.append(train_values.tolist() + [clk_ratio] + coeffs.tolist())
            print(features)
            break

        columns = ["medianKHz", "avSPL", "AvPRF", "Clk_ratio", "Coeff_3", "Coeff_2", "Coeff_1",
                   "Coeff_0"]
        return pd.DataFrame(features, columns=columns)

    def __len__(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self.features)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """Returns a tensor of features for a given index."""
        row = self.features.iloc[idx]
        return torch.tensor(row.values, dtype=torch.float32)


class ClickDataset(Dataset):
    def __init__(self, data_dir, transform=None, resample_interval="10ms", click_filter=None):
        self.transform = transform
        self.resample_interval = resample_interval
        self.data = []
        self.click_filter = click_filter
        self._load_data(data_dir)

    def _load_data(self, data_dir):
        for file in os.listdir(data_dir):
            if file.endswith(".csv"):
                df = pd.read_csv(os.path.join(data_dir, file))
                if self.click_filter:
                    df = self.click_filter(df)
                df["Datetime"] = datetime(1899, 12, 30) + pd.to_timedelta(df["Minute"], unit="m") + pd.to_timedelta(
                    df["microsec"], unit="us")
                df = df[['ICI', 'Datetime']].copy()
                df['Clk/s'] = (1 / df['ICI']) * 1e6
                df.drop(columns=['ICI'], inplace=True)

                df['Datetime'] = pd.to_datetime(df['Datetime'])
                df.set_index('Datetime', inplace=True)

                clk_values = df['Clk/s'].values
                is_inf = np.isinf(clk_values)
                split_indices = np.where(is_inf)[0]

                start_idx = 0
                for end_idx in split_indices:
                    instance = df.iloc[start_idx:end_idx]

                    instance = instance.resample(self.resample_interval).mean().interpolate()

                    if len(instance) > 2:
                        instance.index = (instance.index - instance.index[0]).total_seconds()
                        self.data.append(instance)
                    start_idx = end_idx + 1

        if self.transform:
            self.data = [
                pd.DataFrame({'Clk/s': np.round(self.transform.fit_transform(instance[['Clk/s']]).flatten(), 4)},
                             index=instance.index) for instance in self.data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


dataset = ClickTrainDataset("Click_details", "Train_details")
# print(dataset.train_data.head(5))
print(dataset.__len__())
print(dataset.__getitem__(1))
