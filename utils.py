import pandas as pd
import numpy as np
from pathlib import Path
from fastdtw import fastdtw
import random
from scipy.spatial.distance import euclidean


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
