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
        return self.train_data.reset_index(drop=True)

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
        print("Processing training data...")
        self.train_data = DataProcessor.process_train_data(DataProcessor.read_data(train_folder))
        print("Processing click data...")
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
        print("Matching data...")
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
        print("Extracting features...")
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
        print("Processing data...")
        self.transform = transform
        self.resample_interval = resample_interval
        self.data = []
        self.click_filter = click_filter
        self._load_data(data_dir)

    def _load_data(self, data_dir):
        for file in os.listdir(data_dir):
            df = pd.read_csv(os.path.join(data_dir, file), sep="\t")
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




class ClickDatasetImproved(Dataset):
    def __init__(self, data_dir, target_length=100, click_filter=None, transform=None):
        print("Processing data...")
        self.target_length = target_length  # Fixed sequence length
        self.transform = transform
        self.click_filter = click_filter
        self.data, self.timestamps, self.file_names = self._load_data(data_dir)  # Store file names as well

    def get_by_datetime(self, target_datetime, target_file):
        """Retrieve all instances that match the given datetime and file name up to the second."""
        matching_indices = [
            idx for idx, (timestamp, file_name) in enumerate(zip(self.timestamps, self.file_names))
            if timestamp.replace(microsecond=0) == target_datetime.replace(microsecond=0)
               and file_name == target_file
        ]

        if matching_indices:
            return [self.data[idx] for idx in matching_indices], matching_indices  # Return all matching instances
        return None, []  # Return None if no matches are found

    def _load_data(self, data_dir):
        data_list = []
        timestamps_list = []  # List to store the start timestamps
        file_names_list = []  # List to store corresponding file names

        for file in os.listdir(data_dir):
            df = pd.read_csv(os.path.join(data_dir, file), sep="\t", usecols=['Minute', 'microsec', 'ICI'])

            # Compute absolute timestamps
            df["Datetime"] = datetime(1899, 12, 30) + pd.to_timedelta(df["Minute"], unit="m") + pd.to_timedelta(
                df["microsec"], unit="us")
            df["Clk/s"] = (1 / df["ICI"]) * 1e6  # Convert ICI to clicks per second

            df.drop(columns=['Minute', 'microsec', 'ICI'], inplace=True)
            df['Datetime'] = pd.to_datetime(df['Datetime'])
            df.set_index('Datetime', inplace=True)

            clk_values = df['Clk/s'].values
            is_inf = np.isinf(clk_values)
            split_indices = np.where(is_inf)[0]

            start_idx = 0
            for end_idx in split_indices:
                instance = df.iloc[start_idx:end_idx]

                if len(instance) < 3:  # Skip short sequences
                    start_idx = end_idx + 1
                    continue

                # Apply filtering condition before resampling
                if self.click_filter and not self.click_filter(instance):
                    start_idx = end_idx + 1
                    continue

                # Resample uniformly
                instance = instance.resample("10ms").mean().interpolate()
                if instance.isnull().all().any():
                    start_idx = end_idx + 1
                    continue

                # Normalize time
                time_values = (instance.index - instance.index[0]).total_seconds().astype(np.float32)
                clk_values = instance['Clk/s'].values.astype(np.float32)

                # Ensure valid time values for interpolation
                if len(time_values) < 2 or np.all(time_values == time_values[0]):
                    start_idx = end_idx + 1
                    continue

                # Interpolate to fixed length
                new_time_values = np.linspace(0, time_values[-1], self.target_length, dtype=np.float32)
                interp_func = interp1d(time_values, clk_values, kind='linear', fill_value="extrapolate")
                new_clk_values = interp_func(new_time_values)

                data_list.append(new_clk_values)  # Store as NumPy array

                # Store the start timestamp of the instance
                timestamps_list.append(instance.index[0])

                # Extract and store the first two words of the file name
                file_words = os.path.basename(file).split(' ')[:2]  # Adjust based on actual filename format
                file_names_list.append('_'.join(file_words))

                start_idx = end_idx + 1

        # Convert list of NumPy arrays to a single NumPy array
        data_array = np.stack(data_list, axis=0) if data_list else np.array([])

        # Apply transformation if needed
        if self.transform and data_list:
            data_array = self.transform.fit_transform(data_array)

        return data_array, timestamps_list, file_names_list  # Return both the data, timestamps, and file names

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.timestamps[idx], self.file_names[idx]  # Return the data, timestamp, and file name


# Define the click filter function
def filter_low_click_rate(instance):
    return instance['Clk/s'].max() >= 100  # Keep instances with mean Clk/s >= 50

# dataset = PolynomialDataset("Click_details", "Train_details")
# # print(dataset.train_data.head(5))
# print(dataset.__len__())
# print(dataset.__getitem__(1))
