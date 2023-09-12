import numpy as np
from sklearn.linear_model import BayesianRidge, Lasso
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import cross_val_score
import pickle
from utils.data import *
import argparse
from matplotlib import pyplot as plt
from matplotlib import ticker


class Dataset:
    def __init__(self):
        pass


class MCMCImputer:
    def __init__(
        self,
        lag_length,
        base_model=BayesianRidge,
        param_names=None,
        role_names=None,
    ):
        # Data preparation hyperparameters
        self.lag_length = lag_length
        # Base model for imputation
        if base_model not in [BayesianRidge, Lasso, KNeighborsRegressor, KernelRidge]:
            raise ValueError(
                "Base model must be one of BayesianRidge, Lasso, KNeighborsRegressor, or KernelRidge"
            )
        self.base_classifier = base_model
        self.trained = False

        # Store names of HRV parameters and roles for indexing purposes
        if param_names is None:
            self.param_names = ["PNS index", "SNS index", "Mean RR", "RMSSD", "LF-HF"]
        else:
            self.param_names = param_names

        if role_names is None:
            self.role_names = ["Anes", "Nurs", "Perf", "Surg"]
        else:
            self.role_names = role_names

        self.param_indices = {param: i for i, param in enumerate(self.param_names)}
        self.role_indices = {role: i for i, role in enumerate(self.role_names)}

        # Store imputation models
        self.imputation_models = {}
        for key in param_names:
            self.imputation_models[key] = {}
            for role in self.role_names:
                self.imputation_models[key][role] = self.base_classifier()

    def load(self, filename):
        self.imputation_models = pickle.load(open(filename, "rb"))
        self.trained = True

    def save(self, filename):
        if not self.trained:
            raise ValueError("Imputation models are not trained, cannot save models")
        pickle.dump(self.imputation_models, open(filename, "wb"))

    def impute(self, dataset, burn_in, max_iter, logging_freq=10, verbose=True):
        if not self.trained:
            raise ValueError("Imputation models are not trained, cannot impute data")
        print(f"Begin imputation for {max_iter} iterations and {burn_in} burn-in")
        imputed_dataset = {}
        means = self.get_means(dataset)  # Shape is (num_params, num_actors)
        variances = self.get_stddevs(dataset)  # Shape is (num_params, num_actors)
        # Object for collecting samples
        samples = {}
        for case_idx, case_data in dataset.items():
            samples[case_idx] = {}
            for param_idx, param_data in enumerate(case_data):
                seq_len = param_data.shape[-1]
                param = self.param_names[param_idx]
                # Shape is (num_cases, num_actors, num_timesteps)
                samples[case_idx][param] = np.zeros(
                    (
                        len(self.role_names),
                        seq_len - self.lag_length,
                        max_iter,
                    )
                )

        # Impute missing values in the dataset
        for case_idx, case_data in tqdm(dataset.items()):
            imputed_dataset[case_idx] = np.copy(case_data)
            for param_idx, param_data in enumerate(imputed_dataset[case_idx]):
                # Shape of the data is (num_actors, num_features, num_timesteps)
                for i in range(param_data.shape[-1] - self.lag_length):
                    # Take L timesteps as input vector
                    input_vector = param_data[:, :, i : i + self.lag_length]
                    # For the first L timesteps, use the dataset mean and variance to impute
                    # TODO - Find a way to avoid using means!
                    if i < self.lag_length:
                        for t in range(self.lag_length):
                            mean_impute_idx = np.where(np.isnan(input_vector[:, 0, t]))
                            if len(mean_impute_idx[0] > 1):
                                input_vector[mean_impute_idx, 0, t] = means[
                                    param_idx, mean_impute_idx
                                ]
                    # Take the next timestep as output vector
                    output_vector = param_data[:, :, i + self.lag_length]
                    prev_output_vector = np.zeros(output_vector.shape)
                    # Note indices where imputation is necessary
                    out_impute_idx = np.where(np.isnan(output_vector[:, 0]))
                    if len(out_impute_idx[0] > 1):
                        # Initialize with the mean, then Gibbs sample with imputation models
                        output_vector[out_impute_idx, 0] = means[
                            param_idx, out_impute_idx
                        ]
                        # Gibbs sampling for predicting missing values
                        for t in range(max_iter + burn_in):
                            # Sample each actor's HRV value from its conditional distribution
                            for actor_idx in out_impute_idx[0]:
                                imputed_actor = self.role_names[actor_idx]
                                imputed_param = self.param_names[param_idx]
                                # Add HRV features from remaining actors
                                remaining_actors_data = np.delete(
                                    output_vector[:, 0].reshape(-1), actor_idx, axis=0
                                )
                                input_vector_for_actor = np.append(
                                    input_vector[:, 0, :].reshape(-1),
                                    remaining_actors_data.reshape(-1),
                                )
                                # Add temporal features
                                input_vector_for_actor = np.append(
                                    input_vector_for_actor,
                                    output_vector[actor_idx][1:].reshape(-1),
                                )
                                # Predict missing value
                                input_vector_for_actor = input_vector_for_actor.reshape(
                                    1, -1
                                )
                                imputed_value = self.imputation_models[imputed_param][
                                    imputed_actor
                                ].predict(input_vector_for_actor)

                                prev_output_vector[actor_idx] = output_vector[actor_idx]
                                output_vector[actor_idx] = imputed_value
                                # Store samples after burn-in
                                if t > burn_in:
                                    samples[case_idx][param][
                                        actor_idx, i, t - burn_in
                                    ] = imputed_value[0]

                            # Log progress
                            if t % logging_freq == 0 and verbose:
                                if t < burn_in:
                                    print(f"Case {case_idx} Timestep {i} Burn-in {t}")
                                else:
                                    print(
                                        f"Case {case_idx} Timestep {i} Iteration {t - burn_in}"
                                    )
                            # Check if values have converged and stop MCMC if so
                            if (
                                np.allclose(prev_output_vector, output_vector)
                                and t > burn_in
                            ):
                                # Pad the rest of the samples with the last value
                                for j in range(t - burn_in, max_iter):
                                    for actor_idx in range(output_vector.shape[0]):
                                        samples[case_idx][imputed_param][
                                            actor_idx, i, j
                                        ] = output_vector[actor_idx, 0]
                                break
                    # Assign imputed values to the dataset
                    imputed_dataset[case_idx][param_idx][
                        :, :, i + self.lag_length
                    ] = output_vector
            # For good measure, make sure there are no NaNs left in the dataset
            assert np.all(~np.isnan(imputed_dataset[case_idx]))
        if verbose:
            print("Imputation applied to HRV dataset")

        return imputed_dataset, samples

    def impute_from_first_observed(
        self, dataset, burn_in, max_iter, logging_freq=10, verbose=True
    ):
        if not self.trained:
            raise ValueError("Imputation models are not trained, cannot impute data")
        print(f"Begin imputation for {max_iter} iterations and {burn_in} burn-in")
        imputed_dataset = {}
        means = self.get_means(dataset)  # Shape is (num_params, num_actors)
        # Object for collecting samples
        samples = {}
        for case_idx, case_data in dataset.items():
            if case_idx in [1, 2, 10, 12, 15, 17, 21, 25, 27, 29, 37]:
                continue
            samples[case_idx] = {}
            for param_idx, param_data in enumerate(case_data):
                seq_len = param_data.shape[-1]
                param = self.param_names[param_idx]
                # Shape is (num_actors, num_timesteps, num_samples)
                samples[case_idx][param] = np.zeros(
                    (
                        len(self.role_names),
                        seq_len - self.lag_length,
                        max_iter,
                    )
                )

        # Impute missing values in the dataset
        for case_idx, case_data in tqdm(dataset.items()):
            imputed_dataset[case_idx] = np.copy(case_data)
            for param_idx, param_data in enumerate(imputed_dataset[case_idx]):
                param = self.param_names[param_idx]
                # Find the first set of L timesteps with no missing values and use that as the input vector
                # We will skip all timesteps before
                input_vector = None
                start_idx = None
                for j in range(param_data.shape[-1] - self.lag_length):
                    input_vector = param_data[:, :, j : j + self.lag_length]
                    if np.all(~np.isnan(input_vector[:, 0, :])):
                        start_idx = j
                        break
                # If there is no contiguous set of L timesteps with no missing values, skip this case
                if start_idx is None:
                    print(
                        f"No contiguous set of {self.lag_length} timesteps found in Case {case_idx} {param}"
                    )
                else:
                    # Shape of the data is (num_actors, num_features, num_timesteps)
                    for i in range(start_idx, param_data.shape[-1] - self.lag_length):
                        # Take L timesteps as input vector
                        input_vector = param_data[:, :, i : i + self.lag_length]
                        # Take the next timestep as output vector
                        output_vector = param_data[:, :, i + self.lag_length]
                        prev_output_vector = np.zeros(output_vector.shape)
                        # Note indices where imputation is necessary
                        out_impute_idx = np.where(np.isnan(output_vector[:, 0]))
                        if len(out_impute_idx[0] > 1):
                            # Initialize with the mean, then Gibbs sample with imputation models
                            output_vector[out_impute_idx, 0] = means[
                                param_idx, out_impute_idx
                            ]
                            # Gibbs sampling for predicting missing values
                            for t in range(max_iter + burn_in):
                                # Sample each actor's HRV value from its conditional distribution
                                for actor_idx in out_impute_idx[0]:
                                    imputed_actor = self.role_names[actor_idx]
                                    imputed_param = self.param_names[param_idx]
                                    # Add HRV features from remaining actors
                                    remaining_actors_data = np.delete(
                                        output_vector[:, 0].reshape(-1),
                                        actor_idx,
                                        axis=0,
                                    )
                                    input_vector_for_actor = np.append(
                                        input_vector[:, 0, :].reshape(-1),
                                        remaining_actors_data.reshape(-1),
                                    )
                                    # Add temporal features
                                    input_vector_for_actor = np.append(
                                        input_vector_for_actor,
                                        output_vector[actor_idx][1:].reshape(-1),
                                    )
                                    # Predict missing value
                                    input_vector_for_actor = (
                                        input_vector_for_actor.reshape(1, -1)
                                    )
                                    imputed_value = self.imputation_models[
                                        imputed_param
                                    ][imputed_actor].predict(input_vector_for_actor)

                                    prev_output_vector[actor_idx] = output_vector[
                                        actor_idx
                                    ]
                                    output_vector[actor_idx] = imputed_value
                                    # Store samples after burn-in
                                    if t > burn_in:
                                        samples[case_idx][param][
                                            actor_idx, i, t - burn_in
                                        ] = imputed_value[0]

                                # Log progress
                                if t % logging_freq == 0 and verbose:
                                    if t < burn_in:
                                        print(
                                            f"Case {case_idx} Timestep {i} Burn-in {t}"
                                        )
                                    else:
                                        print(
                                            f"Case {case_idx} Timestep {i} Iteration {t - burn_in}"
                                        )
                                # Check if values have converged and stop MCMC if so
                                if (
                                    np.allclose(prev_output_vector, output_vector)
                                    and t > burn_in
                                ):
                                    # Pad the rest of the samples with the last value
                                    for j in range(t - burn_in, max_iter):
                                        for actor_idx in range(output_vector.shape[0]):
                                            samples[case_idx][imputed_param][
                                                actor_idx, i, j
                                            ] = output_vector[actor_idx, 0]
                                    break
                        # Assign imputed values to the dataset
                        imputed_dataset[case_idx][param_idx][
                            :, :, i + self.lag_length
                        ] = output_vector
                    # For good measure, make sure there are no NaNs left in the dataset after the starting index
                    assert np.all(
                        ~np.isnan(imputed_dataset[case_idx][:, :, start_idx:])
                    )
        if verbose:
            print("Imputation applied to HRV dataset")

        return imputed_dataset, samples

    def get_means(self, dataset):
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

    def get_stddevs(self, dataset):
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

    def train(self, dataset, verbose=True):
        # Create input-output pairs for each parameter
        input_output_pairs = self.create_input_output_pairs(dataset)
        # Train imputation models for each parameter
        for param in self.param_names:
            for role in self.role_names:
                X, y = self.get_individual_features(input_output_pairs, param, role)
                self.imputation_models[param][role].fit(X, y)
                # Cross validation
                score = cross_val_score(
                    self.imputation_models[param][role],
                    X,
                    y,
                    cv=5,
                    scoring="neg_mean_squared_error",
                )
                if verbose:
                    print(
                        f"Cross-validation MSE for imputing {param} {role} => Mean: {-score.mean():.4f} Max: {-score.min():.4f} Min: {-score.max():.4f}"
                    )
        self.trained = True
        self.save("./models/imputation_models.pkl")

    def get_individual_features(self, input_output_pairs, param, role):
        # Create training data by concatenating remaining features to the existing input vector
        X = []
        y = []
        actor_idx = self.role_indices[role]
        for sample in input_output_pairs[param]:
            target = sample[1][actor_idx][0]
            remaining_actors_data = np.delete(sample[1][:, 0].reshape(-1), actor_idx)
            input_vector = np.append(
                sample[0][:, 0, :].reshape(-1),
                remaining_actors_data.reshape(-1),
            )
            # Add temporal features
            input_vector = np.append(input_vector, sample[1][actor_idx][1:].reshape(-1))
            X.append(input_vector)
            y.append(target)
        return np.array(X), np.array(y)

    def create_input_output_pairs(self, dataset):
        input_output_pairs = {}
        for key in self.param_names:
            input_output_pairs[key] = []
        for case_idx, case_data in dataset.items():
            # Shape of the data is (num_params, num_actors, num_features, num_timesteps)
            for param_idx, param_data in enumerate(case_data):
                # Shape of the data is (num_actors, num_features, num_timesteps)
                # Timestep axis is moved to the front for broadcasting
                num_timesteps = param_data.shape[-1]
                param_data = np.transpose(param_data, (2, 0, 1))
                # Check if there are any NaNs in any feature for any of the actors
                # Shape of the mask is (num_timesteps,)
                mask = np.all(~np.isnan(param_data), axis=(1, 2))
                param_data = np.transpose(param_data, (1, 2, 0))
                # Use the mask to check if there are any missing values in the last L + 1 timesteps
                for i in range(num_timesteps - self.lag_length - 1):
                    if np.all(mask[i : i + self.lag_length + 1]):
                        # If there are no missing values, add to training set
                        # The first L steps are used as input and the last one is used as output
                        key = self.param_names[param_idx]
                        input_vector = param_data[:, :, i : i + self.lag_length]
                        output_vector = param_data[:, :, i + self.lag_length]
                        assert np.all(~np.isnan(output_vector))
                        input_output_pairs[key].append((input_vector, output_vector))
        return input_output_pairs


def imputation_line_plot(
    unimputed_dataset,
    imputed_dataset,
    time_interval=5,
    data_dir="./data",
    plots_dir="./plots",
    first_phase_removed=True,
):
    cases_summary = pd.read_excel(f"{data_dir}/NIH-OR-cases-summary.xlsx").iloc[1:, :]
    phase_ids = get_phase_ids(time_interval=time_interval)
    limits = {
        "PNS index": [-4, 5],
        "SNS index": [-2, 12],
        "Mean RR": [400, 1200],
        "RMSSD": [0, 160],
        "LF-HF": [0, 70],
    }
    for case_id, unimputed_case_data in unimputed_dataset.items():
        for param_id, unimputed_param_data in enumerate(unimputed_case_data):
            # Get corresponding data from imputed dataset
            assert case_id in imputed_dataset
            imputed_param_data = imputed_dataset[case_id][param_id][:, 0, :]
            unimputed_param_data = unimputed_dataset[case_id][param_id][:, 0, :]
            assert unimputed_param_data.shape == imputed_param_data.shape

            plt.figure(figsize=(10, 4))
            n_samples = unimputed_param_data.shape[1]
            x_axis = np.linspace(0, n_samples - 1, n_samples)
            for actor_id, unimputed_role_data in enumerate(unimputed_param_data):
                imputed_role_data = imputed_param_data[actor_id]
                role = role_names[actor_id]
                actor_name = cases_summary.loc[cases_summary["Case"] == case_id][
                    role
                ].values[0]
                # actor_param_mean = means[param_id, actor_id]
                # plt.axhline(
                #     y=actor_param_mean, color=f"C{actor_id}", linestyle="--", alpha=0.4
                # )
                # Plot imputed data
                plt.plot(
                    x_axis,
                    imputed_role_data,
                    color=f"C{actor_id}",
                    linestyle="--",
                    alpha=0.6,
                )
                # Superimpose unimputed data
                plt.plot(
                    x_axis,
                    unimputed_role_data,
                    label=f"{role} {actor_name}",
                    color=f"C{actor_id}",
                )

            # Add phase ID lines
            minor_ticks = []
            minor_labels = []
            offset = 0
            for phase, interval in enumerate(phase_ids[case_id]):
                if first_phase_removed and phase == 0:
                    offset = phase_ids[case_id][phase][1]
                    continue
                if interval[0] is not None and interval[1] is not None:
                    plt.axvspan(
                        interval[0] - offset,
                        interval[1] - offset,
                        color="black",
                        alpha=0.1,
                    )
                    midpoint = (
                        int(interval[0] + (interval[1] - interval[0]) / 2) - offset
                    )
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

            plt.title(
                f"Imputed HRV for case {case_id}: {param_names[param_id]} ({time_interval}min)"
            )
            plt.xlabel("Timestep")
            plt.ylabel(param_names[param_id])
            # plt.ylim(limits[param_names[param_id]])
            plt.legend()

            # Write plots to file
            if not os.path.exists(f"{plots_dir}/imputation/Case{case_id:02d}"):
                os.makedirs(f"{plots_dir}/imputation/Case{case_id:02d}")
            plt.savefig(
                f"{plots_dir}/imputation/Case{case_id:02d}/{param_names[param_id]}.png"
            )

            plt.close()


def mcmc_plot(
    samples,
    original_dataset,
    imputed_dataset,
    time_interval=5,
    data_dir="./data",
    lag_length=3,
    plots_dir="./plots",
    first_phase_removed=True,
    actor_names=["Anes", "Nurs", "Perf", "Surg"],
    param_names=["PNS index", "SNS index", "Mean RR", "RMSSD", "LF-HF"],
):
    cases_summary = pd.read_excel(f"{data_dir}/NIH-OR-cases-summary.xlsx").iloc[1:, :]
    phase_ids = get_phase_ids(time_interval=time_interval)
    limits = {
        "PNS index": [-4, 5],
        "SNS index": [-2, 12],
        "Mean RR": [400, 1200],
        "RMSSD": [0, 160],
        "LF-HF": [0, 70],
    }
    for case_id, per_case_samples in tqdm(samples.items()):
        for param_name, param_samples in per_case_samples.items():
            # Shape of the data is (num_actors, seq_len - lag_length, num_iterations)
            for actor_idx in range(param_samples.shape[0]):
                # Create line plot by taking the mean of the samples at every timestep
                actor_name = actor_names[actor_idx]
                param_idx = param_names.index(param_name)
                actor_samples = param_samples[actor_idx]
                plt.figure(figsize=(10, 4))
                n_samples = actor_samples.shape[0]
                # Collect sample means and stddevs per timestep
                original_samples = original_dataset[case_id][param_idx, actor_idx, 0, :]
                imputed_samples = imputed_dataset[case_id][param_idx, actor_idx, 0, :]
                seq_len = original_samples.shape[0]
                n_samples = actor_samples.shape[0]
                start_idx = original_samples.shape[0] - n_samples
                sample_means = np.mean(actor_samples, axis=1)
                sample_stddevs = np.std(actor_samples, axis=1)

                # Add phase ID lines
                minor_ticks = []
                minor_labels = []
                offset = 0
                for phase, interval in enumerate(phase_ids[case_id]):
                    if first_phase_removed and phase == 0:
                        offset = phase_ids[case_id][phase][1]
                        continue
                    if interval[0] is not None and interval[1] is not None:
                        plt.axvspan(
                            interval[0] - offset,
                            interval[1] - offset,
                            color="black",
                            alpha=0.1,
                        )
                        midpoint = (
                            int(interval[0] + (interval[1] - interval[0]) / 2) - offset
                        )
                        minor_ticks.append(midpoint)
                        minor_labels.append(f"P{phase+1}")

                # Add x-axis markers for phase IDs
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

                # Plot sample means as a line and stddevs as an errorbar
                # Starting from start_idx
                x_axis = np.linspace(start_idx, seq_len - 1, n_samples)
                plt.plot(
                    x_axis,
                    sample_means,
                    label="sample mean",
                    color=f"C{actor_idx}",
                )
                plt.fill_between(
                    x_axis,
                    sample_means - sample_stddevs,
                    sample_means + sample_stddevs,
                    alpha=0.2,
                    color=f"C{actor_idx}",
                )

                x_axis = np.linspace(0, seq_len - 1, seq_len)

                # Plot imputed data
                plt.plot(
                    x_axis,
                    imputed_samples,
                    label=f"imputed",
                    color=f"C{actor_idx}",
                    linestyle="--",
                    alpha=0.4,
                )
                # Plot original data
                plt.plot(
                    x_axis,
                    original_samples,
                    label=f"original",
                    color=f"C{actor_idx}",
                    alpha=0.6,
                )

                # Save plot
                plt.title(
                    f"MCMC samples for case {case_id}: {param_name} ({time_interval}min)"
                )
                plt.xlabel("Timestep")
                plt.ylabel(param_name)
                plt.legend()

                # Write plots to file
                if not os.path.exists(f"{plots_dir}/mcmc/Case{case_id:02d}"):
                    os.makedirs(f"{plots_dir}/mcmc/Case{case_id:02d}")
                plt.savefig(
                    f"{plots_dir}/mcmc/Case{case_id:02d}/{param_name}_{actor_name}.png"
                )

                plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--time_interval", type=int, default=5)
    parser.add_argument("--drop_first", type=bool, default=False)
    parser.add_argument("--plot", type=bool, default=True)
    parser.add_argument("--normalize_per_case", type=bool, default=False)
    parser.add_argument("--standardize", type=bool, default=False)
    parser.add_argument("--temporal_features", type=bool, default=True)
    parser.add_argument("--static_features", type=bool, default=False)
    parser.add_argument("--burn_in", type=int, default=100)
    parser.add_argument("--max_iter", type=int, default=100)
    parser.add_argument("--lag_length", type=int, default=3)
    parser.add_argument("--verbose", type=bool, default=False)
    parser.add_argument("--logging_freq", type=int, default=10)
    parser.add_argument("--save_dataset", type=bool, default=True)
    parser.add_argument("--train", type=bool, default=True)
    args = parser.parse_args()

    param_names = ["PNS index", "SNS index", "Mean RR", "RMSSD", "LF-HF"]
    role_names = ["Anes", "Nurs", "Perf", "Surg"]

    # Load dataset
    dataset = make_dataset_from_file(
        data_dir=args.data_dir,
        time_interval=args.time_interval,
        temporal_features=args.temporal_features,
        static_features=args.static_features,
        standardize=args.standardize,
    )

    if args.drop_first:
        dataset = drop_edge_phases(dataset)
    if args.normalize_per_case:
        dataset = normalize_per_case(dataset)

    if args.train:
        # Impute data
        imputer = MCMCImputer(
            lag_length=args.lag_length, param_names=param_names, role_names=role_names
        )
        imputer.train(dataset, verbose=args.verbose)
        imputed_dataset, mcmc_samples = imputer.impute(
            dataset=dataset,
            burn_in=args.burn_in,
            max_iter=args.max_iter,
            logging_freq=args.logging_freq,
            verbose=args.verbose,
        )
        if args.save_dataset:
            pickle.dump(
                imputed_dataset, open(f"{args.data_dir}/imputed_dataset.pkl", "wb")
            )
            pickle.dump(mcmc_samples, open(f"{args.data_dir}/mcmc_samples.pkl", "wb"))
    else:
        imputed_dataset = pickle.load(
            open(f"{args.data_dir}/imputed_dataset.pkl", "rb")
        )
        mcmc_samples = pickle.load(open(f"{args.data_dir}/mcmc_samples.pkl", "rb"))

    # Plot imputed data
    if args.plot:
        imputation_line_plot(
            dataset,
            imputed_dataset,
            time_interval=args.time_interval,
            first_phase_removed=args.drop_first,
        )
        mcmc_plot(
            mcmc_samples,
            dataset,
            imputed_dataset,
            time_interval=args.time_interval,
            lag_length=args.lag_length,
            first_phase_removed=args.drop_first,
        )

    # when doing cross validation, make a scatterplot
    # refactor training to have a separate cross validation function for this
    # also make an imputation line plot using MCMC samples like GP plots
