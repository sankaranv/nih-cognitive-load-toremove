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
import datetime
import matplotlib.ticker as ticker
from utils.data import *


def plot_params(
    dataset,
    means,
    phase_ids,
    case_id,
    plots_dir,
    latex_dir,
    time_interval="5min",
    normalize=False,
):
    if normalize:
        limits = {
            "PNS index": [-6, 6],
            "SNS index": [-6, 6],
            "Mean RR": [-6, 6],
            "RMSSD": [-6, 6],
            "LF-HF": [-6, 6],
        }
    else:
        limits = {
            "PNS index": [-4, 5],
            "SNS index": [-2, 12],
            "Mean RR": [400, 1200],
            "RMSSD": [0, 160],
            "LF-HF": [0, 70],
        }
    for case_data in dataset:
        for param_id, param_data in enumerate(case_data):
            plt.figure(figsize=(10, 4))
            n_samples = param_data.shape[1]
            x_axis = np.linspace(0, n_samples - 1, n_samples)
            for actor_id, role_data in enumerate(param_data):
                role = role_names[actor_id]
                actor_name = cases_summary.loc[cases_summary["Case"] == case_id][
                    role
                ].values[0]
                actor_param_mean = means[param_id, actor_id]
                plt.axhline(
                    y=actor_param_mean, color=f"C{actor_id}", linestyle="--", alpha=0.4
                )
                if normalize:
                    role_mean = np.nanmean(role_data, axis=-1)
                    role_std = np.nanstd(role_data, axis=-1)
                    role_data = (role_data - role_mean) / role_std
                plt.plot(x_axis, role_data, label=f"{role} {actor_name}")

            # Add phase ID lines
            minor_ticks = []
            minor_labels = []
            for phase, interval in enumerate(phase_ids[case_id]):
                plt.axvspan(interval[0], interval[1], color="black", alpha=0.1)
                if interval[0] is not None and interval[1] is not None:
                    midpoint = int(interval[0] + (interval[1] - interval[0]) / 2)
                    minor_ticks.append(midpoint)
                    minor_labels.append(f"P{phase+1}")

            ax = plt.gca()
            ax.xaxis.set_minor_locator(ticker.FixedLocator(minor_ticks))
            ax.xaxis.set_minor_formatter(ticker.FixedFormatter(minor_labels))
            ax.set_xticks(minor_ticks, minor=True)
            ax.tick_params(
                axis="x",
                which="minor",
                direction="out",
                top=True,
                labeltop=True,
                bottom=False,
                labelbottom=False,
            )
            if normalize:
                plt.title(
                    f"Heart rate variability for case {case_id}: {param_names[param_id]} ({time_interval})"
                )
            else:
                plt.title(
                    f"Heart rate variability for case {case_id}: {param_names[param_id]} ({time_interval}, normalized)"
                )
            plt.xlabel("Timestep")
            plt.ylabel(param_names[param_id])
            plt.ylim(limits[param_names[param_id]])
            plt.legend()

            # Write plots to file
            # if not os.path.isdir(f'{plots_dir}/line_plots/Case{case_id:02d}'):
            #     os.mkdir(f'{plots_dir}/line_plots/Case{case_id:02d}')
            if normalize:
                plot_type_dirname = "normalized_line_plots"
            else:
                plot_type_dirname = "line_plots"
            plt.savefig(
                f"{plots_dir}/{time_interval}/{plot_type_dirname}/Case{case_id:02d}/{param_names[param_id]}.png"
            )

            # Push plots to latex document
            if latex_dir is not None:
                plt.savefig(
                    f"{latex_dir}/plots/{time_interval}/{plot_type_dirname}/Case{case_id:02d}/{param_names[param_id]}.png"
                )

            plt.close()


def generate_line_plots(
    plots_dir="plots", latex_dir=None, time_interval="5min", normalize=False
):
    means = get_means(time_interval, normalize)
    phase_ids = get_phase_ids(time_interval=time_interval)
    print(means)
    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39]:
            try:
                dataset = import_case_data(case_id=i, time_interval=time_interval)
                plot_params(
                    dataset,
                    means,
                    phase_ids,
                    i,
                    plots_dir,
                    latex_dir,
                    time_interval,
                    normalize,
                )
            except Exception as e:
                print(e)


def plot_densities_by_role(latex_dir=None, time_interval="5min"):
    # Collect data by actor ID
    dataset_by_actor = {"Anes": {}, "Nurs": {}, "Perf": {}, "Surg": {}}
    for i in range(1, 41):
        if i not in [5, 9, 14, 16, 24, 39]:
            dataset = import_case_data(case_id=i, time_interval=time_interval)
            for param_id, param_data in enumerate(dataset[0]):
                for actor_id, role_data in enumerate(param_data):
                    samples = role_data[np.where(~np.isnan(role_data))]
                    role = role_names[actor_id]
                    param = param_names[param_id]
                    key = cases_summary.loc[cases_summary["Case"] == i][role].values[0]
                    if param not in dataset_by_actor[role]:
                        dataset_by_actor[role][param] = {}
                    if key in dataset_by_actor[role][param]:
                        dataset_by_actor[role][param][key] = np.hstack(
                            (dataset_by_actor[role][param][key], samples)
                        )
                    elif len(samples) > 0:
                        dataset_by_actor[role][param][key] = role_data
    for role in role_names:
        for param in param_names:
            if not os.path.isdir(f"plots/{time_interval}/density_plots/{role}"):
                os.mkdir(f"plots/{time_interval}/density_plots/{role}")
            plt.figure(figsize=(10, 4))
            for actor, samples in dataset_by_actor[role][param].items():
                sns.set_style("whitegrid")
                sns.kdeplot(samples, bw_method=0.5, label=actor)
                plt.xlabel(f"{param}")
                plt.ylabel("Density")
                plt.title(f"Density of {param} ({time_interval}) for {role}")
                plt.legend()
                print(f"{role} {param} {actor}")
            plt.savefig(f"plots/{time_interval}/density_plots/{role}/{param}.png")
            if latex_dir is not None:
                plt.savefig(
                    f"{latex_dir}/plots/{time_interval}/density_plots/{role}/{param}.png"
                )
            plt.close()


def generate_per_phase_density_plots(latex_dir=None, time_interval="5min"):
    per_phase_normalized_samples = {
        "PNS index": {},
        "SNS index": {},
        "Mean RR": {},
        "RMSSD": {},
        "LF-HF": {},
    }
    phase_ids = get_phase_ids(time_interval=time_interval)
    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39, 28]:
            try:
                dataset = import_case_data(case_id=i, time_interval=time_interval)[0]
                # Ignore cases with missing per-step data
                if dataset.shape[-1] > 1:
                    means = np.nanmean(dataset, axis=-1)
                    std = np.nanstd(dataset, axis=-1)
                    dataset = (dataset - means[:, :, None]) / std[:, :, None]
                    for param_id, param_name in enumerate(param_names):
                        for actor_id, role_name in enumerate(role_names):
                            if (
                                role_name
                                not in per_phase_normalized_samples[param_name].keys()
                            ):
                                per_phase_normalized_samples[param_name][role_name] = {}
                            for phase, interval in enumerate(phase_ids[i]):
                                if interval[0] is not None and interval[1] is not None:
                                    per_case_samples = dataset[
                                        param_id, actor_id, interval[0] : interval[1]
                                    ]
                                    if (
                                        phase
                                        not in per_phase_normalized_samples[param_name][
                                            role_name
                                        ].keys()
                                    ):
                                        per_phase_normalized_samples[param_name][
                                            role_name
                                        ][phase] = per_case_samples
                                    else:
                                        per_phase_normalized_samples[param_name][
                                            role_name
                                        ][phase] = np.concatenate(
                                            (
                                                per_phase_normalized_samples[
                                                    param_name
                                                ][role_name][phase],
                                                per_case_samples,
                                            )
                                        )
            except Exception as e:
                print(e)

    for role in role_names:
        for param in param_names:
            for phase_id, per_phase_samples in per_phase_normalized_samples[param][
                role
            ].items():
                samples = per_phase_samples[~np.isnan(per_phase_samples)]
                sns.set_style("whitegrid")
                sns.kdeplot(
                    per_phase_samples,
                    bw_method=0.5,
                    label=f"Phase {phase_id} ({len(per_phase_samples)} samples)",
                )
                plt.xlabel(f"Normalized {param}")
                plt.ylabel("Density")
                plt.title(
                    f"Per-phase density of normalized {param} ({time_interval}) for {role}"
                )
            plt.legend()
            plt.savefig(
                f"plots/{time_interval}/density_plots/per_phase/{param}/{role}.png"
            )
            if latex_dir is not None:
                plt.savefig(
                    f"{latex_dir}/plots/{time_interval}/density_plots/per_phase/{param}/{role}.png"
                )
            plt.close()


def generate_scatterplots(latex_dir=None, time_interval="5min"):
    means = get_means(time_interval)

    # Collect correlation coefficients for each parameter
    param_corr_coef_samples = {
        "PNS index": [],
        "SNS index": [],
        "Mean RR": [],
        "RMSSD": [],
        "LF-HF": [],
    }

    for i in tqdm(range(1, 41)):
        if i not in [5, 9, 14, 16, 24, 39]:
            try:
                dataset = import_case_data(case_id=i, time_interval=time_interval)[0]
                # Ignore cases with missing per-step data
                if dataset.shape[-1] > 1:
                    means = np.nanmean(dataset, axis=-1)
                    std = np.nanstd(dataset, axis=-1)
                    dataset = (dataset - means[:, :, None]) / std[:, :, None]
                    for param_id, param_name in enumerate(param_names):
                        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
                        for ax_counter, (idx_1, idx_2) in enumerate(
                            combinations([0, 1, 2, 3], 2)
                        ):
                            samples_x = dataset[param_id, idx_1, :]
                            samples_y = dataset[param_id, idx_2, :]

                            # # Set ranges for scatterplots
                            # if (not np.isnan(samples_x).all()) and (not np.isnan(samples_y).all()):
                            #     range_high = ceil(max(max(samples_x), max(samples_y)))
                            #     range_low = ceil(min(min(samples_x), min(samples_y)))
                            #     range_equal = max(abs(range_high), abs(range_low))
                            # else:
                            #     range_equal = 3
                            range_equal = 6

                            # Correlation coefficient that is NaN-sensitive
                            corr_coef = np.ma.corrcoef(
                                np.ma.masked_invalid(samples_x),
                                np.ma.masked_invalid(samples_y),
                            )
                            if corr_coef[0, 1].data == 0:
                                corr_coef = corr_coef[0, 1]
                            else:
                                corr_coef = corr_coef[0, 1]
                                param_corr_coef_samples[param_name].append(corr_coef)

                            # Plot
                            axs.flat[ax_counter].scatter(samples_x, samples_y)
                            axs.flat[ax_counter].set_xlabel(
                                f"{role_names[idx_1]} {param_name}"
                            )
                            axs.flat[ax_counter].set_ylabel(
                                f"{role_names[idx_2]} {param_name}"
                            )
                            axs.flat[ax_counter].set_xlim(-range_equal, range_equal)
                            axs.flat[ax_counter].set_ylim(-range_equal, range_equal)
                            axs.flat[ax_counter].set_title(f"rho = {corr_coef:04f}")

                        # Save overall plot
                        fig.suptitle(f"Standardized {param_name} for Case {i:02d}")
                        fig.savefig(
                            f"plots/{time_interval}/scatterplots/{param_name}/Case{i:02d}.png"
                        )
                        if latex_dir is not None:
                            fig.savefig(
                                f"{latex_dir}/plots/{time_interval}/scatterplots/{param_name}/Case{i:02d}.png"
                            )
                        plt.close()
            except Exception as e:
                print(e)

    # Density plot of correlation coefficients
    for param_name in param_names:
        sns.set_style("whitegrid")
        sns.kdeplot(
            param_corr_coef_samples[param_name], bw_method=0.5, label=param_name
        )
        plt.xlabel(f"Pearson Correlation Coefficient")
        plt.ylabel("Density")
        plt.title(
            f"Density of correlation coefficient ({time_interval}) between pairs of actors"
        )
    plt.legend()
    plt.savefig(f"plots/{time_interval}/density_plots/corr_coef.png")
    if latex_dir is not None:
        plt.savefig(f"{latex_dir}/plots/{time_interval}/density_plots/corr_coef.png")
    plt.close()


latex_dir = "648bad436055ba2df65649bc"
# generate_per_phase_density_plots(latex_dir, time_interval='1min')
generate_line_plots("plots", latex_dir, time_interval="5min", normalize=True)
# generate_scatterplots(latex_dir, time_interval='5min')
# plot_densities_by_role(latex_dir, time_interval='5min')
