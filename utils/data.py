import pandas as pd
import csv
import itertools
import numpy as np
import os
from tqdm import tqdm
from itertools import combinations
from math import ceil
import datetime
import random
import torch

global param_names, param_indices
global role_names, role_indices
global cases_summary

param_names = ["PNS index", "SNS index", "Mean RR", "RMSSD", "LF-HF"]
param_indices = {"PNS index": 0, "SNS index": 1, "Mean RR": 2, "RMSSD": 3, "LF-HF": 4}
role_names = ["Anes", "Nurs", "Perf", "Surg"]
role_indices = {"Anes": 0, "Nurs": 1, "Perf": 2, "Surg": 3}
cases_summary = pd.read_excel("./data/NIH-OR-cases-summary.xlsx").iloc[1:, :]


def import_case_data(data_dir="./data", case_id=3, time_interval="5min"):
    relevant_lines = [66, 67, 72, 78, 111]
    if time_interval == "5min":
        phase_name = "cognitiveLoad-phases-5min"
    else:
        phase_name = "cognitiveLoad-phases-1min"
    dataset = []
    max_num_samples = 0
    for role_name in role_names:
        if time_interval == "5min":
            file_name = f"{data_dir}/Case{case_id:02d}/{phase_name}/3296_{case_id:02d}_{role_name}_hrv.csv"
        else:
            file_name = f"{data_dir}/Case{case_id:02d}/{phase_name}/3296_{case_id:02d}_{role_name}_hrv-1min.csv"
        if not os.path.isfile(file_name):
            if max_num_samples > 0:
                empty_role_data = np.full((5, max_num_samples), np.nan)
            else:
                empty_role_data = np.full((5, 1), np.nan)
            dataset.append(empty_role_data)
        else:
            role_data = []
            with open(file_name, "r") as f:
                r = csv.reader(f)
                for i in itertools.count(start=1):
                    if i > relevant_lines[-1]:
                        break
                    elif i not in relevant_lines:
                        next(r)
                    else:
                        try:
                            row = next(r)
                            row = [x.replace(" ", "") for x in row]
                            row = [x for x in row if x != ""][1:]
                            row = [float(x) if x != "NaN" else np.nan for x in row]
                            role_data.append(row)
                        except StopIteration as e:
                            print("End of file reached")
            role_data[-1] = role_data[-1][::2]
            role_data = np.array(role_data)
            dataset.append(role_data)
            if role_data.shape[1] > max_num_samples:
                max_num_samples = role_data.shape[1]

    # Add padding to the data for missing samples
    # This assumes all measurements start at the same time and just cut off early for some roles!
    for i, role_data in enumerate(dataset):
        if role_data.shape[1] < max_num_samples:
            pad_length = max_num_samples - role_data.shape[1]
            empty_data = np.full((5, pad_length), np.nan)
            dataset[i] = np.hstack((role_data, empty_data))

    dataset = np.array(dataset)
    dataset = np.expand_dims(dataset, axis=0)
    dataset = np.swapaxes(dataset, 1, 2)
    return dataset


def get_means(data_dir="./data", time_interval="5min"):
    means = np.zeros((5, 4))
    num_samples = np.zeros((5, 4))
    for i in range(1, 41):
        if i not in [5, 9, 14, 16, 24, 39]:
            try:
                dataset = import_case_data(
                    data_dir=data_dir, case_id=i, time_interval=time_interval
                )[0]
                num_samples += np.sum(~np.isnan(dataset), axis=-1)
                means += np.sum(np.nan_to_num(dataset), axis=-1)
            except Exception as e:
                print(f"There was a problem with case {i}")
    return means / num_samples


def get_per_phase_actor_ids(data_dir="./data", time_interval="5min"):
    per_phase_actor_ids = {"Anes": {}, "Nurs": {}, "Perf": {}, "Surg": {}}
    phase_ids = get_phase_ids(time_interval=time_interval)
    print("Getting per phase actor IDs")
    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39, 28]:
            try:
                dataset = import_case_data(
                    data_dir=data_dir, case_id=i, time_interval=time_interval
                )[0]
                # Ignore cases with missing per-step data
                if dataset.shape[-1] > 1:
                    # We assume all parameters have the same number of measurements in the dataset
                    param_id = 0
                    for actor_id, role_name in enumerate(role_names):
                        for phase, interval in enumerate(phase_ids[i]):
                            if interval[0] is not None and interval[1] is not None:
                                per_case_samples = dataset[
                                    param_id, actor_id, interval[0] : interval[1]
                                ]
                                # Log actor names in case needed for coloring scatterplots
                                # Obtain actor name for the given case
                                case_actor_name = cases_summary.loc[
                                    cases_summary["Case"] == i
                                ][role_name].values[0]
                                if phase not in per_phase_actor_ids[role_name]:
                                    per_phase_actor_ids[role_name][phase] = np.full(
                                        len(per_case_samples), case_actor_name
                                    )
                                else:
                                    per_phase_actor_ids[role_name][
                                        phase
                                    ] = np.concatenate(
                                        (
                                            per_phase_actor_ids[role_name][phase],
                                            np.full(
                                                len(per_case_samples), case_actor_name
                                            ),
                                        )
                                    )
            except Exception as e:
                print(e)
    return per_phase_actor_ids


def hms_to_min(s):
    t = 0
    for u in s.split(":"):
        t = 60 * t + int(u)
    # Round to the minute
    if t % 60 >= 30:
        return int(t / 60) + 1
    else:
        return int(t / 60)


def get_phase_ids(data_dir="./data", time_interval="5min"):
    phase_ids = {}
    for i in range(1, 41):
        if i not in [5, 9, 14, 16, 24, 39]:
            phase_ids[i] = []
            file_name = f"{data_dir}/Case{i:02d}/3296_{i:02d}-abstractedPhases.csv"
            with open(file_name, "r") as f:
                df = pd.read_csv(file_name, header=0)
                df = df.dropna(axis=0, how="all")
                for phase, row in df.iterrows():
                    start_time = row["Onset_Time"]
                    stop_time = row["Offset_Time"]
                    if not isinstance(start_time, str):
                        phase_ids[i].append([None, None])
                    else:
                        start_idx = hms_to_min(start_time) // int(time_interval[0])
                        stop_idx = hms_to_min(stop_time) // int(time_interval[0])
                        phase_ids[i].append([start_idx, stop_idx])
    return phase_ids


def make_dataset_from_file(data_dir="./data", time_interval="5min", param_id=None):
    # Shape of the data for each case is (5, 4, num_samples) or (4, num_samples)
    dataset = {}
    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39, 28]:
            if param_id is None:
                dataset[i] = import_case_data(
                    data_dir=data_dir, case_id=i, time_interval=time_interval
                )[0]
            else:
                dataset[i] = import_case_data(
                    data_dir=data_dir, case_id=i, time_interval=time_interval
                )[0][param_id]
    return dataset


def make_nan_masks(dataset):
    nan_masks = {}
    for case_id in dataset.keys():
        nan_masks[case_id] = np.isnan(dataset[case_id])
    return nan_masks


def get_mask(data):
    if isinstance(data, np.ndarray):
        return np.isnan(data)
    elif isinstance(data, torch.Tensor):
        return torch.isnan(data)


def get_max_len(dataset):
    max_len = 0
    for case_id in dataset.keys():
        if dataset[case_id].shape[-1] > max_len:
            max_len = dataset[case_id].shape[-1]
    return max_len


def make_train_test_split(
    dataset,
    train_split: float = 0.6,
    val_split: float = 0.2,
    test_split: float = 0.2,
):
    cases = list(dataset.keys())
    random.shuffle(cases)
    if train_split + val_split + test_split != 1:
        raise ValueError(
            "Train, validation, and test splits must sum to 1.0. "
            f"Current splits sum to {train_split + val_split + test_split}"
        )
    train_idx = int(train_split * len(cases))
    val_idx = train_idx + int(val_split * len(cases))
    train_cases = cases[:train_idx]
    val_cases = cases[train_idx:val_idx]
    test_cases = cases[val_idx:]
    train_dataset = {case: dataset[case] for case in train_cases}
    val_dataset = {case: dataset[case] for case in val_cases}
    test_dataset = {case: dataset[case] for case in test_cases}
    return train_dataset, val_dataset, test_dataset


class HRVDataset(torch.utils.data.Dataset):
    """Holds HRV data in torch.Tensor format
    Dataset is indexed by case ID and returns a (data, nan_mask) tuple
    Each element in the tuple has shape (4, seq_len)
    """

    def __init__(self, dataset):
        self.cases = list(dataset.keys())
        self.data = self.make_tensor_from_dict(dataset)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def make_tensor_from_dict(self, dataset):
        for case_id in dataset.keys():
            dataset[case_id] = torch.from_numpy(dataset[case_id]).float()
        return dataset


def get_input_output_sequences(
    input_window: int, output_window: int, hrv_dataset: HRVDataset
):
    """Prepares batches for transformer model training
    Returns a sequence of pairs of tensors with shape (4, input_window) and (4, output_window)

    Args:
        input_window (int): number of time steps to include in the input sequence
        output_window (int): number of time steps to include in the output sequence
        hrv_dataset (HRVDataset): a dataset of HRV parameter values
    """
    in_out_seq = []
    for i in hrv_dataset.cases:
        data = hrv_dataset[i]
        n = data.shape[-1]
        for j in range(n - input_window - output_window):
            # Slice out input sequence and take the next (output_window) values as the output sequence
            in_out_seq.append(
                (
                    data[:, j : j + input_window],
                    data[:, j + input_window : j + input_window + output_window],
                )
            )
    return in_out_seq


def get_batch(in_out_seq, batch_size, start_idx=0):
    """Returns a batch of input/output sequences from a list of input/output sequences

    Args:
        in_out_seq_data (list): a list of tuples of input/output sequences
        in_out_seq_mask (list): a list of tuples of input/output nan masks
        batch_size (int): the number of sequences to include in the batch
        start_idx (int): the index of the first sequence to include in the batch.
    """
    num_actors = in_out_seq[0][0].shape[0]
    input_window = in_out_seq[0][0].shape[1]
    output_window = in_out_seq[0][1].shape[1]
    n = min(batch_size, len(in_out_seq) - start_idx)
    batched_input_data = torch.Tensor(num_actors, input_window, n)
    batched_output_data = torch.Tensor(num_actors, output_window, n)
    for i in range(n):
        batched_input_data[:, :, i] = in_out_seq[start_idx + i][0]
        batched_output_data[:, :, i] = in_out_seq[start_idx + i][1]
    return batched_input_data, batched_output_data


if __name__ == "__main__":
    # Load dataset
    dataset = HRVDataset(make_dataset_from_file(param_id=1))
    # Create input-output sequences
    in_out_seq = get_input_output_sequences(10, 3, dataset)
    print(len(in_out_seq), in_out_seq[0][0].shape, in_out_seq[0][1].shape)
    # Create batches
    batched_input_data, batched_output_data = get_batch(in_out_seq, 32)
    print(batched_input_data.shape, batched_output_data.shape)
