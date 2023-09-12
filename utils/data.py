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
from typing import Union

global param_names, param_indices
global temporal_feature_names, static_feature_names
global role_names, role_indices

param_names = ["PNS index", "SNS index", "Mean RR", "RMSSD", "LF-HF"]
temporal_feature_names = ["Phase ID", "Time"]
static_feature_names = None
param_indices = {"PNS index": 0, "SNS index": 1, "Mean RR": 2, "RMSSD": 3, "LF-HF": 4}
role_names = ["Anes", "Nurs", "Perf", "Surg"]
role_indices = {"Anes": 0, "Nurs": 1, "Perf": 2, "Surg": 3}


def import_case_data(
    data_dir="./data", case_id=3, time_interval=5, pad_to_max_len=False, max_length=None
):
    relevant_lines = [66, 67, 72, 78, 111]
    if time_interval == 5:
        phase_name = "cognitiveLoad-phases-5min"
    elif time_interval == 1:
        phase_name = "cognitiveLoad-phases-1min"
    else:
        raise ValueError(
            f"Data is only available for intervals of 1min or 5min, not {time_interval}"
        )
    dataset = []
    max_num_samples = 0
    for role_name in role_names:
        if time_interval == 5:
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
        if pad_to_max_len:
            if max_length is None:
                raise ValueError(
                    "Must provide max_length if padding to max length for all cases"
                )
            padding_length = max_length
        else:
            padding_length = max_num_samples
        if role_data.shape[1] < padding_length:
            pad_length = max_num_samples - role_data.shape[1]
            empty_data = np.full((5, pad_length), np.nan)
            dataset[i] = np.hstack((role_data, empty_data))

    dataset = np.array(dataset)
    dataset = np.expand_dims(dataset, axis=0)
    dataset = np.swapaxes(dataset, 1, 2)
    return dataset


def get_means(dataset):
    means = np.zeros((5, 4))
    num_samples = np.zeros((5, 4))
    for _, case_data in dataset.items():
        if len(case_data.shape) == 3:
            data = case_data
        elif case_data.shape[-2] > 1:
            # If temporal or static features are present, ignore them
            data = case_data[:, :, 0, :]
        num_samples += np.sum(~np.isnan(data), axis=-1)
        means += np.sum(np.nan_to_num(data), axis=-1)
    return means / num_samples


def get_stddevs(dataset):
    samples = {}
    std_devs = np.zeros((5, 4))
    for _, data in dataset.items():
        for x in range(5):
            for y in range(4):
                if (x, y) not in samples:
                    samples[(x, y)] = np.array([])
                samples[(x, y)] = np.concatenate(
                    (samples[(x, y)], data[x][y][~np.isnan(data[x][y])])
                )
    for x in range(5):
        for y in range(4):
            std_devs[x, y] = np.std(samples[(x, y)])
    return std_devs


def get_per_phase_actor_ids(dataset, time_interval=5, data_dir="./data"):
    per_phase_actor_ids = {"Anes": {}, "Nurs": {}, "Perf": {}, "Surg": {}}
    phase_ids = get_phase_ids(time_interval=time_interval)
    cases_summary = pd.read_excel(f"{data_dir}/NIH-OR-cases-summary.xlsx").iloc[1:, :]
    print("Getting per phase actor IDs")
    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39, 28]:
            try:
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


def get_phase_ids(data_dir="./data", time_interval=5):
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
                        start_idx = hms_to_min(start_time) // time_interval
                        stop_idx = hms_to_min(stop_time) // time_interval
                        phase_ids[i].append([start_idx, stop_idx])
    return phase_ids


def make_dataset_from_file(
    data_dir="./data",
    time_interval=5,
    param_id=None,
    standardize=False,
    max_length=None,
    pad_to_max_len=False,
    temporal_features=False,
    static_features=False,
):
    # Shape of the data for each case is (5, 4, num_samples) or (4, num_samples)
    dataset = {}
    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39, 28]:
            dataset[i] = import_case_data(
                data_dir=data_dir,
                case_id=i,
                time_interval=time_interval,
                max_length=max_length,
                pad_to_max_len=pad_to_max_len,
            )[0]

    # Standardize dataset
    if standardize:
        means = get_means(dataset)
        std_devs = get_stddevs(dataset)
        for case_id in dataset.keys():
            for x in range(5):
                for y in range(4):
                    dataset[case_id][x, y] -= means[x, y]
                    dataset[case_id][x, y] /= std_devs[x, y]

    if param_id is not None:
        for case_id in dataset.keys():
            dataset[case_id] = dataset[case_id][param_id]

    # Add temporal features if requested
    if temporal_features:
        dataset = add_temporal_features(dataset)

    # Add static features if requested
    if static_features:
        dataset = add_static_features(dataset)

    return dataset


def get_nan_mask(data):
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

    return (train_dataset, val_dataset, test_dataset)


def make_temporal_features(
    dataset, time_interval=5, num_phases=8, return_type="np", pad_phase_on=True
):
    """Make temporal features for time-series models
    We will use phase ID and time within the surgery as features

    Args:
        dataset (dict): a dictionary of HRV parameter values
        time_interval (str): the time interval to use for temporal features
        num_phases (int): the number of phases to use for temporal features
        return_type (str): Specifies whether to return a numpy array or torch tensor. Defaults to "np".
    """

    phase_ids = get_phase_ids(time_interval=time_interval)
    temporal_features = {}

    for case in range(1, 41):
        if case not in [5, 9, 14, 16, 24, 39, 28]:
            # For every case, create a vector of features for each time step
            # Set the last feature to be the time within the surgery
            if return_type == "np":
                features = np.zeros((num_phases + 2, dataset[case].shape[-1]))
                features[-1, :] = np.array([j for j in range(dataset[case].shape[-1])])
            else:
                features = torch.zeros((num_phases + 2, dataset[case].shape[-1]))
                features[-1, :] = torch.Tensor(
                    [j for j in range(dataset[case].shape[-1])]
                )

            # The remaining features are a one-hot vector indicating which phase is active at each time step
            # The last phase indicates that no phase is active
            features[:-1, num_phases + 1] = 1

            for phase in range(num_phases):
                phase_start = phase_ids[case][phase][0]
                phase_end = phase_ids[case][phase][1]
                if phase_end is not None and phase_start is not None:
                    if pad_phase_on:
                        # The feature is always 1 for any timestep after the phase starts
                        features[phase, phase_start:] = 1
                    else:
                        # The feature is 1 only when the phase is active and 0 when it finishes
                        features[phase, phase_start:phase_end] = 1
                    # A phase was active so set the no phase feature to 0
                    features[num_phases + 1, phase_start:phase_end] = 0

            temporal_features[case] = features

            # # Stack features for each of the four actors
            # if return_type == "np":
            #     temporal_features[case] = np.stack(
            #         [features] * dataset[case].shape[0], axis=0
            #     )
            # elif return_type == "torch":
            #     temporal_features[case] = torch.stack(
            #         [features] * dataset[case].shape[0], axis=0
            #     )

    return temporal_features


def one_hot(
    idx: Union[int, list],
    num_categories: int,
    zero_based=True,
    return_type: str = "list",
):
    """Create a one-hot vector for the given categorical feature
    Supports both zero-based and one-based indexing, and multiclass features

    Args:
        idx (int, list): the index or list of indices of the categories
        num_categories (int): the number of categories
        zero_based (bool): whether the index is zero-based or one-based
        return_type (str): Specifies whether to return a numpy array, torch tensor, or list. Defaults to "list".

    Returns:
        list: a one-hot vector
    """
    # Create vector of zeros
    if return_type == "list":
        one_hot = [0] * num_categories
    elif return_type == "np":
        one_hot = np.zeros(num_categories)
    elif return_type == "torch":
        one_hot = torch.zeros(num_categories)

    # Assign value 1 to the given index
    if isinstance(idx, int):
        if zero_based:
            one_hot[idx] = 1
        else:
            one_hot[idx - 1] = 1
        return one_hot
    elif (
        isinstance(idx, list) or isinstance(idx, np.ndarray) or isinstance(torch.Tensor)
    ):
        for i in idx:
            if zero_based:
                one_hot[i] = 1
            else:
                one_hot[i - 1] = 1
        return one_hot

    else:
        raise ValueError(f"Indices should be of type int or list, got {type(idx)}")


def make_static_features(
    data_dir="./data", return_type="np", unroll_through_time=False, lengths=None
):
    """Make static features for time-series models
    We will use procedure type, number of vessels, 30-day mortality,
    180-day mortality, 30-day morbidity, 30-day SSI, and the IDs of the
    anesthesiologist, perfusionist, surgeon, and nurse

    Args:
        data_dir (str): the directory containing the data
        return_type (str): Specifies whether to return a numpy array or torch Tensor. Defaults to "np".

    Returns:
        dict: a dictionary of static features
    """
    static_features = {}
    surg_procedure_encoding = {"CABG": 0, "AVR": 1, "min. inv. AVR": 2, "AVR/CABG": 3}
    metadata = pd.read_excel(
        f"{data_dir}/metadata-for-statisticians-2022-10-04.xlsx", header=1
    )
    for idx, row in metadata.iterrows():
        case_id = int(row["Case ID"].split("_")[1])
        if case_id not in [5, 9, 14, 16, 24, 39, 28]:
            procedure_type = surg_procedure_encoding[row["Procedure Type"]]
            no_vessels = 0 if pd.isna(row["No. Vessels"]) else int(row["No. Vessels"])
            day_mort_30 = round(row["30 Day Mort."] * 100, 2)
            day_mort_180 = round(row["180 Day Mort."] * 100, 2)
            day_morb_30 = round(row["30 Day Morb."] * 100, 2)
            day_ssi_30 = round(row["30 Day SSI"] * 100, 2)
            anes_id = int(row["Anesthesia Code"][-2:])
            perf_id = int(row["Perfusionist Code"][-2:])
            surg_id = int(row["Surgeon Code"][-2:])
            nurs_id = int(row["Nurse Code"][-2:])
            features = (
                one_hot(procedure_type, 4)
                + [
                    no_vessels,
                    day_mort_30,
                    day_mort_180,
                    day_morb_30,
                    day_ssi_30,
                ]
                + one_hot(anes_id, 5, zero_based=False)
                + one_hot(perf_id, 5, zero_based=False)
                + one_hot(surg_id, 3, zero_based=False)
                # + one_hot(nurs_id, 18) # 18 categories is very long and some are never in the data, skipping this feature for now
            )
            if return_type == "np":
                static_features[case_id] = np.array(features)
            elif return_type == "torch":
                static_features[case_id] = torch.Tensor(features)
            else:
                raise ValueError(
                    f"Invalid return type {return_type}. Must be 'np' or 'torch'"
                )

    if unroll_through_time:
        if lengths is None:
            raise ValueError(
                "Lengths must be provided if unrolling static features through time"
            )
        return unroll_static_features(static_features, lengths, return_type)
    else:
        return static_features


def get_lengths(dataset):
    lengths = {}
    for case_id in dataset.keys():
        lengths[case_id] = dataset[case_id].shape[-1]
    return lengths


def unroll_static_features(static_features, lengths, return_type="np"):
    static_feature_dict = {}
    for case_id in lengths.keys():
        if return_type == "np":
            static_feature_dict[case_id] = np.tile(
                static_features[case_id], (lengths[case_id], 1)
            ).transpose()
        elif return_type == "torch":
            static_feature_dict[case_id] = torch.tile(
                static_features[case_id], (lengths[case_id], 1)
            ).transpose()
        else:
            raise ValueError(
                f"Invalid return type {return_type}. Must be 'np' or 'torch'"
            )
    return static_feature_dict


def add_temporal_features(dataset):
    # Look at the first key in the dataset to determine the return type
    if isinstance(dataset[list(dataset.keys())[0]], np.ndarray):
        return_type = "np"
    elif isinstance(dataset[list(dataset.keys())[0]], torch.Tensor):
        return_type = "torch"
    else:
        raise ValueError(
            f"Invalid dataset type {type(dataset)}. Must be np.ndarray or torch.Tensor"
        )

    # Make the temporal features
    temporal_features = make_temporal_features(dataset, return_type=return_type)

    # Temporal features are identical for every parameter and actor
    # Dataset shape is (num_params, num_actors, num_samples)
    # Temporal features shape is (num_temporal_features, num_samples)
    # The returned dataset should have shape (num_params, num_actors, num_temporal_features + 1, num_samples)
    new_dataset = {}

    for case_id in dataset.keys():
        case_data = dataset[case_id]

        # If dataset shape is (num_params, num_actors, num_samples)
        # Add an axis such that the shape is now (num_params, num_actors, 1, num_samples)
        if len(case_data.shape) == 3:
            if return_type == "np":
                case_data = np.expand_dims(case_data, axis=2)
            else:
                case_data = case_data.unsqueeze(2)

        # Dataset shape should be (num_params, num_actors, num_features, num_samples)
        if return_type == "np":
            new_dataset[case_id] = np.concatenate(
                (
                    case_data,
                    np.tile(
                        temporal_features[case_id],
                        (case_data.shape[0], case_data.shape[1], 1, 1),
                    ),
                ),
                axis=2,
            )
        elif return_type == "torch":
            new_dataset[case_id] = torch.cat(
                (
                    case_data,
                    torch.tile(
                        temporal_features[case_id],
                        (case_data.shape[0], case_data.shape[1], 1, 1),
                    ),
                ),
                dim=2,
            )

    return new_dataset


def add_static_features(dataset):
    # Look at the first key in the dataset to determine the return type
    if isinstance(dataset[list(dataset.keys())[0]], np.ndarray):
        return_type = "np"
    elif isinstance(dataset[list(dataset.keys())[0]], torch.Tensor):
        return_type = "torch"
    else:
        raise ValueError(
            f"Invalid dataset type {type(dataset)}. Must be np.ndarray or torch.Tensor"
        )

    # Make the static features
    static_features = make_static_features(
        return_type=return_type, unroll_through_time=True, lengths=get_lengths(dataset)
    )

    # Static features are identical for every parameter and actor
    # Dataset shape can be (num_params, num_actors, num_samples) or (num_params, num_actors, num_features, num_samples)
    # Temporal features shape is (num_static_features, num_samples)
    # The returned dataset should have shape (num_params, num_actors, num_features + num_static_features, num_samples)

    new_dataset = {}

    for case_id in dataset.keys():
        case_data = dataset[case_id]

        # If dataset shape is (num_params, num_actors, num_samples)
        # Add an axis such that the shape is now (num_params, num_actors, 1, num_samples)
        if len(case_data.shape) == 3:
            if return_type == "np":
                case_data = np.expand_dims(case_data, axis=2)
            else:
                case_data = case_data.unsqueeze(2)

        # Dataset shape should be (num_params, num_actors, num_features, num_samples)
        if return_type == "np":
            new_dataset[case_id] = np.concatenate(
                (
                    case_data,
                    np.tile(
                        static_features[case_id],
                        (case_data.shape[0], case_data.shape[1], 1, 1),
                    ),
                ),
                axis=2,
            )
        elif return_type == "torch":
            new_dataset[case_id] = torch.cat(
                (
                    case_data,
                    torch.tile(
                        static_features[case_id],
                        (case_data.shape[0], case_data.shape[1], 1, 1),
                    ),
                ),
                dim=2,
            )

    return new_dataset


def combine_feature_sets(feature_dicts: tuple, keys: list, return_type="np"):
    """Combine feature vectors from the different dictionaries of feature sets
    Used to append any combination of static, temporal, and HRV features

    Args:
        feature_dicts (tuple): a tuple of dictionaries of feature vectors
    """
    output_dict = {}
    for feature_dict in feature_dicts:
        if not isinstance(feature_dict, dict):
            raise ValueError(
                f"Expected a dictionary of features but got {type(feature_dict)}"
            )
        for key in keys:
            if key not in feature_dict:
                raise ValueError(f"Key {key} not found in feature dictionary")
            if key not in output_dict:
                output_dict[key] = []
            output_dict[key].append(feature_dict[key])

    for key in keys:
        if return_type == "np":
            output_dict[key] = np.hstack(output_dict[key])
        elif return_type == "torch":
            output_dict[key] = torch.cat(output_dict[key], dim=0)
        else:
            raise ValueError(
                f"Invalid return type {return_type}. Must be 'np' or 'torch'"
            )

    return output_dict


def extract_fully_observed_sequences(dataset):
    """
    Extract fully observed sequences from dataset
    """
    num_cases = 0
    fully_observed_sequences = None
    for case_idx, case_data in dataset.items():
        usable_rows_per_case = None
        num_cases += case_data.shape[-1]
        for param_data in case_data:
            param_data = np.transpose(param_data, (2, 0, 1))
            mask = np.all(~np.isnan(param_data), axis=(1, 2))
            usable_rows_per_param = param_data[mask]
            usable_rows_per_param = np.transpose(usable_rows_per_param, (1, 2, 0))
            usable_rows_per_param = usable_rows_per_param[np.newaxis, :, :, :]
            if usable_rows_per_case is None:
                usable_rows_per_case = usable_rows_per_param
            else:
                usable_rows_per_case = np.concatenate(
                    (usable_rows_per_case, usable_rows_per_param), axis=0
                )
        if fully_observed_sequences is None:
            fully_observed_sequences = usable_rows_per_case
        else:
            fully_observed_sequences = np.concatenate(
                (fully_observed_sequences, usable_rows_per_case), axis=3
            )

    # Report percentage of usable cases
    print(
        f"{fully_observed_sequences.shape[-1]/num_cases * 100 :.2f}% of the data is fully observed"
    )
    print(f"Shape of fully observed sequence data: {fully_observed_sequences.shape}")
    return fully_observed_sequences


def extract_all_sequences(dataset):
    sequences = None
    for case_idx, case_data in dataset.items():
        print(case_data.shape)
        if sequences is None:
            sequences = case_data
        else:
            sequences = np.concatenate((sequences, case_data), axis=3)
    return sequences


def drop_edge_phases(dataset, drop_first=True, drop_last=False, time_interval=5):
    """Drop all timesteps from the first phase and/or last phase of each case
       Hopefully this gets rid of the long empty sequences at the start of prediction window

    Args:
        dataset (dict): dataset of HRV parameters
    """
    phase_ids = get_phase_ids(time_interval=time_interval)
    first_key = next(iter(phase_ids))
    first_phase_id = 0
    last_phase_id = len(phase_ids[first_key]) - 1
    new_dataset = {}
    for case_id in dataset.keys():
        # Case data has shape (num_params, num_actors, num_features, num_samples)
        case_data = dataset[case_id]
        first_phase_end_idx = phase_ids[case_id][first_phase_id][1]
        last_phase_start_idx = phase_ids[case_id][last_phase_id][0]
        if drop_first and first_phase_end_idx is not None:
            new_dataset[case_id] = case_data[:, :, :, first_phase_end_idx:]
        if drop_last and last_phase_start_idx is not None:
            new_dataset[case_id] = case_data[:, :, :, :last_phase_start_idx]
    return new_dataset


def normalize_per_case(dataset):
    """Normalize each case to have mean 0 and std 1"""
    new_dataset = {}
    for case_id in dataset.keys():
        case_data = dataset[case_id]
        means = np.nanmean(case_data, axis=-1)
        stds = np.nanstd(case_data, axis=-1)
        new_dataset[case_id] = (case_data - means[..., np.newaxis]) / stds[
            ..., np.newaxis
        ]
    return new_dataset


if __name__ == "__main__":
    dataset = make_dataset_from_file(temporal_features=True, static_features=True)
    for k in dataset.keys():
        print(dataset[k].shape)
