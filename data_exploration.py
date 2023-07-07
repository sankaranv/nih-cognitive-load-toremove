import pandas as pd
import csv
import itertools
import numpy as np
import os 
from matplotlib import pyplot as plt
from tqdm import tqdm
import seaborn as sns
from itertools import combinations
from math import ceil

param_names = ['PNS index', 'SNS index', 'Mean RR', 'RMSSD', 'LF-HF']
role_names = ['Anes', 'Nurs', 'Perf', 'Surg']
cases_summary = pd.read_excel('data/NIH-OR-cases-summary.xlsx').iloc[1:, :]

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

def plot_params(dataset, means, case_id, plots_dir, latex_dir):
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
                role = role_names[actor_id]
                actor_name = cases_summary.loc[cases_summary['Case'] == case_id][role].values[0]
                actor_param_mean = means[param_id, actor_id]
                plt.axhline(y = actor_param_mean, color=f"C{actor_id}", linestyle = '--', alpha = 0.4)
                plt.plot(x_axis, role_data, label=f"{role} {actor_name}")
            plt.title(f"Heart rate variability for case {case_id}: {param_names[param_id]}")
            plt.xlabel("Timestep")
            plt.ylabel(param_names[param_id])
            plt.ylim(limits[param_names[param_id]])
            plt.legend()

            # Write plots to file
            # if not os.path.isdir(f'{plots_dir}/line_plots/Case{case_id:02d}'):
            #     os.mkdir(f'{plots_dir}/line_plots/Case{case_id:02d}')
            plt.savefig(f'{plots_dir}/line_plots/Case{case_id:02d}/{param_names[param_id]}.png')

            # Push plots directly to Overleaf
            if latex_dir is not None:
                plt.savefig(f'{latex_dir}/plots/line_plots/Case{case_id:02d}/{param_names[param_id]}.png')
            
            plt.close()

def generate_line_plots(plots_dir='plots', latex_dir=None):
    means = get_means()
    print(means)
    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39]:
            try:
                dataset = import_case_data(case_id = i)
                plot_params(dataset, means, i, plots_dir, latex_dir)
            except Exception as e:
                print(e)

def get_means():
    means = np.zeros((5, 4))
    num_samples = np.zeros((5, 4))
    for i in range(1, 41):
        if i not in [5, 9, 14, 16, 24, 39]:
            try:
                dataset = import_case_data(case_id = i)[0]
                num_samples += np.sum(~np.isnan(dataset), axis=-1)
                means += np.sum(np.nan_to_num(dataset), axis=-1)
            except Exception as e:
                print(f"There was a problem with case {i}")
    return means / num_samples

def plot_densities_by_role(latex_dir=None):
    # Collect data by actor ID
    dataset_by_actor = {'Anes': {}, 'Nurs': {}, 'Perf': {}, 'Surg': {}}
    for i in range(1, 41):
        if i not in [5, 9, 14, 16, 24, 39]:
            dataset = import_case_data(case_id = i)
            for param_id, param_data in enumerate(dataset[0]):
                for actor_id, role_data in enumerate(param_data):
                    samples = role_data[np.where(~np.isnan(role_data))]
                    role = role_names[actor_id]
                    param = param_names[param_id]
                    key = cases_summary.loc[cases_summary['Case'] == i][role].values[0]
                    if param not in dataset_by_actor[role]:
                        dataset_by_actor[role][param] = {}
                    if key in dataset_by_actor[role][param]:
                        dataset_by_actor[role][param][key] = np.hstack((dataset_by_actor[role][param][key], samples))
                    elif len(samples) > 0:
                        dataset_by_actor[role][param][key] = role_data
    for role in role_names: 
        for param in param_names: 
            if not os.path.isdir(f"plots/density_plots/{role}"):
                os.mkdir(f"plots/density_plots/{role}")
            plt.figure(figsize=(10,4))
            for actor, samples in dataset_by_actor[role][param].items():
                sns.set_style('whitegrid')
                sns.kdeplot(samples, bw_method=0.5, label=actor)
                plt.xlabel(f"{param}")
                plt.ylabel("Density")
                plt.title(f"Density of {param} for {role}")
                plt.legend()
                print(f"{role} {param} {actor}")
            plt.savefig(f"plots/density_plots/{role}/{param}.png")
            if latex_dir is not None:
                plt.savefig(f"{latex_dir}/plots/density_plots/{role}/{param}.png")
            plt.close()

def generate_scatterplots(latex_dir=None):
    means = get_means()
    print(means)
    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39]:
            try:
                dataset = import_case_data(case_id = i)[0]
                # Ignore cases with missing per-step data
                if dataset.shape[-1] > 1:
                    means = np.nanmean(dataset, axis=-1)
                    std = np.nanstd(dataset, axis=-1)
                    dataset = (dataset - means[:, :, None]) / std[:, :, None]
                    for param_id, param_name in enumerate(param_names):
                        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                        for ax_counter, (idx_1, idx_2) in enumerate(combinations([0,1,2,3],2)):
                            samples_x = dataset[param_id, idx_1, :]
                            samples_y = dataset[param_id, idx_2, :]

                            # Set ranges for scatterplots
                            if (not np.isnan(samples_x).all()) and (not np.isnan(samples_y).all()):
                                range_high = ceil(max(max(samples_x), max(samples_y)))
                                range_low = ceil(min(min(samples_x), min(samples_y)))
                                range_equal = max(abs(range_high), abs(range_low))
                            else:
                                range_equal = 3
                            
                            axs.flat[ax_counter].scatter(samples_x, samples_y)
                            axs.flat[ax_counter].set_xlabel(f"{role_names[idx_1]} {param_name}")
                            axs.flat[ax_counter].set_ylabel(f"{role_names[idx_2]} {param_name}")
                            axs.flat[ax_counter].set_xlim(-range_equal, range_equal)
                            axs.flat[ax_counter].set_ylim(-range_equal, range_equal)
                        fig.suptitle(f"Standardized {param_name} for Case {i:02d}")
                        fig.savefig(f"plots/scatterplots/{param_name}/Case{i:02d}.png")
                        if latex_dir is not None:
                            fig.savefig(f"{latex_dir}/plots/scatterplots/{param_name}/Case{i:02d}.png")
                        plt.close()
            except Exception as e:
                print(e)


latex_dir = '648bad436055ba2df65649bc'
generate_line_plots('plots', latex_dir)
generate_scatterplots(latex_dir)
plot_densities_by_role(latex_dir)


'''
Next steps

- Add correlation coefficients to every scatterplots
- Density plot of all correlation coefficients


'''

