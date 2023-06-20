import pandas as pd
import csv
import itertools
import numpy as np
import os 
from matplotlib import pyplot as plt
from tqdm import tqdm

param_names = ['PNS index', 'SNS index', 'Mean RR', 'RMSSD', 'LF-HF']
role_names = ['Anes', 'Nurs', 'Perf', 'Surg']

def import_case_data(data_dir = 'data', case_id = 3):
    relevant_lines = [66, 67, 72, 78, 111]
    phase_name = 'cognitiveLoad-phases-5min'
    dataset = []
    max_num_samples = 0
    for role_name in role_names:
        file_name = f"{data_dir}/Case{case_id:02d}/{phase_name}/3296_{case_id:02d}_{role_name}_hrv.csv"
        if not os.path.isfile(file_name):
            if max_num_samples > 0:
                empty_role_data = np.full((5, max_num_samples), np.nan)
            else:
                empty_role_data = np.full((5, 1), np.nan)
            dataset.append(empty_role_data)
        else:
            role_data = []
            with open(file_name, 'r') as f:
                r = csv.reader(f)
                for i in itertools.count(start=1):
                    if i > relevant_lines[-1]:
                        break
                    elif i not in relevant_lines:
                        next(r)
                    else:
                        try:
                            row = next(r)
                            row = [x.replace(' ', '') for x in row]
                            row = [x for x in row if x!=''][1:]
                            row = [float(x) if x!='NaN' else np.nan for x in row]
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

def plot_params(dataset, case_id):
    limits = {'PNS index': [-4,5],
              'SNS index': [-2,12], 
              'Mean RR': [400,1200], 
              'RMSSD': [0,160], 
              'LF-HF': [0,70] 
              }
    for case_data in dataset:
        for param_id, param_data in enumerate(case_data):
            plt.figure(figsize=(10,4))
            n_samples = param_data.shape[1]
            x_axis = np.linspace(0, n_samples-1, n_samples)
            for actor_id, role_data in enumerate(param_data):
                plt.plot(x_axis, role_data, label=role_names[actor_id])
            plt.title(f"Heart rate variability for case {case_id}: {param_names[param_id]}")
            plt.xlabel("Sample")
            plt.ylabel(param_names[param_id])
            plt.ylim(limits[param_names[param_id]])
            plt.legend()
            if not os.path.isdir(f'plots/Case{case_id:02d}'):
                os.mkdir(f'plots/Case{case_id:02d}')
            plt.savefig(f'plots/Case{case_id:02d}/{param_names[param_id]}.png')
            plt.close()

def generate_all_plots():
    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39]:
            try:
                dataset = import_case_data(case_id = i)
                plot_params(dataset, i)
            except Exception as e:
                print(f"There was a problem with case {i}")

latex_dir = '648bad436055ba2df65649bc'
generate_all_plots()

