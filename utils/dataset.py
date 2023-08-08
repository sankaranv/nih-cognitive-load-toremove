import pandas as pd
import csv
import itertools
import numpy as np
import os
from tqdm import tqdm
from itertools import combinations
from math import ceil
import datetime

global param_names
global role_names
global cases_summary


param_names = ["PNS index", "SNS index", "Mean RR", "RMSSD", "LF-HF"]
role_names = ["Anes", "Nurs", "Perf", "Surg"]
cases_summary = pd.read_excel("data/NIH-OR-cases-summary.xlsx").iloc[1:, :]


def import_case_data(data_dir="data", case_id=3, time_interval="5min"):
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


def get_means(time_interval="5min"):
    means = np.zeros((5, 4))
    num_samples = np.zeros((5, 4))
    for i in range(1, 41):
        if i not in [5, 9, 14, 16, 24, 39]:
            try:
                dataset = import_case_data(case_id=i, time_interval=time_interval)[0]
                num_samples += np.sum(~np.isnan(dataset), axis=-1)
                means += np.sum(np.nan_to_num(dataset), axis=-1)
            except Exception as e:
                print(f"There was a problem with case {i}")
    return means / num_samples


def get_per_phase_actor_ids(time_interval="5min"):
    per_phase_actor_ids = {"Anes": {}, "Nurs": {}, "Perf": {}, "Surg": {}}
    phase_ids = get_phase_ids(time_interval=time_interval)
    print("Getting per phase actor IDs")
    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39, 28]:
            try:
                dataset = import_case_data(case_id=i, time_interval=time_interval)[0]
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


def get_phase_ids(data_dir="data", time_interval="5min"):
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


def make_dataset(time_interval="5min"):
    dataset = {}
    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39, 28]:
            dataset[i] = import_case_data(case_id=i, time_interval="5min")[0]
    return dataset
