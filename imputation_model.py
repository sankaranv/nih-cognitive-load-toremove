from utils.data import *
from sklearn.linear_model import BayesianRidge
import sys
from sklearn.model_selection import cross_val_score
import pickle
from matplotlib import pyplot as plt
from matplotlib import ticker
from sklearn.neighbors import KNeighborsRegressor


def create_imputation_train_test_data(dataset, lag_length=3):
    """Find contiguous sequences of n timesteps with no missing data
        Input vector is the last n fully observed instances, skip if there is missing data in-between
        Output is the timestep with missing data

    Args:
        dataset (dict): HRV dataset
    """
    means = get_means(dataset)  # Shape is (num_params, num_actors)
    train_test_pairs = {}
    for key in param_names:
        train_test_pairs[key] = []
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
            # Use the mask to check if there are any missing values in the last three timesteps
            for i in range(num_timesteps - lag_length):
                if np.all(mask[i : i + lag_length]):
                    # If there are no missing values, add to train set
                    key = param_names[param_idx]
                    input_vector = param_data[:, :, i : i + lag_length]
                    output_vector = param_data[:, :, i + lag_length]
                    # In the output vector we will use the mean to fill in the missing values
                    # Not a great idea, but we will see how it goes
                    mean_impute_idx = np.where(np.isnan(output_vector[:, 0]))
                    if len(mean_impute_idx[0] > 1):
                        output_vector[mean_impute_idx, 0] = means[
                            param_idx, mean_impute_idx
                        ]
                    assert np.all(~np.isnan(output_vector))
                    train_test_pairs[key].append((input_vector, output_vector))
    return train_test_pairs


def train_imputation_models(dataset, lag_length=3, verbose=True):
    train_test_pairs = create_imputation_train_test_data(dataset, lag_length)
    # Multiple Imputation by Chained Equations
    # Predict each feature in the output vector using the remaining features and the input vector
    # Missing values in the output vector were imputed with the mean
    # Input vectors have shape (num_actors, num_features, lag_length)
    # TODO: experiment with different regressors, for now we use BayesianRidge
    imputation_models = {}
    for param in param_names:
        imputation_models[param] = {}
        for actor_idx, actor in enumerate(role_names):
            model = BayesianRidge()
            # Create training data by concatenating remaining features to the existing input vector
            X = []
            y = []
            for sample in train_test_pairs[param]:
                target = sample[1][actor_idx][0]
                remaining_actors_data = np.delete(
                    sample[1][:, 0].reshape(-1), actor_idx
                )
                input_vector = np.append(
                    sample[0][:, 0, :].reshape(-1),
                    remaining_actors_data.reshape(-1),
                )
                # Add temporal features
                input_vector = np.append(
                    input_vector, sample[1][actor_idx][1:].reshape(-1)
                )
                X.append(input_vector)
                y.append(target)
            X = np.array(X)
            y = np.array(y)
            # print(X.shape, y.shape)
            # Cross validation
            score = cross_val_score(model, X, y, cv=5, scoring="neg_mean_squared_error")
            if verbose:
                print(
                    f"Cross-validation MSE for imputing {param} {actor} => Mean: {-score.mean():.4f} Max: {-score.min():.4f} Min: {-score.max():.4f}"
                )
            # Fit the model
            model.fit(X, y)
            # Save model
            imputation_models[param][actor] = model
    pickle.dump(imputation_models, open("checkpoints/imputation_models.pkl", "wb"))
    return imputation_models


def impute_data(dataset, lag_length=3, verbose=True):
    imputation_models = train_imputation_models(dataset, lag_length, verbose)
    means = get_means(dataset)  # Shape is (num_params, num_actors)
    imputed_dataset = {}
    # Impute missing values in the dataset
    for case_idx, case_data in dataset.items():
        imputed_dataset[case_idx] = np.copy(case_data)
        for param_idx, param_data in enumerate(imputed_dataset[case_idx]):
            # Shape of the data is (num_actors, num_features, num_timesteps)
            for i in range(param_data.shape[-1] - lag_length):
                # Take first lag_length timesteps as input vector
                input_vector = param_data[:, :, i : i + lag_length]
                # If there are missing values, impute with the mean
                # This should only matter for the first lag_length timesteps
                for t in range(lag_length):
                    mean_impute_idx = np.where(np.isnan(input_vector[:, 0, t]))
                    if len(mean_impute_idx[0] > 1):
                        input_vector[mean_impute_idx, 0, t] = means[
                            param_idx, mean_impute_idx
                        ]
                output_vector = param_data[:, :, i + lag_length]
                # Note indices where imputation is necessary
                # Temporarily impute with the mean, then use the imputation models to assign final value
                out_impute_idx = np.where(np.isnan(output_vector[:, 0]))
                if len(out_impute_idx[0] > 1):
                    output_vector[out_impute_idx, 0] = means[param_idx, out_impute_idx]
                    # Predict missing values
                    for actor_idx in out_impute_idx[0]:
                        imputed_actor = role_names[actor_idx]
                        imputed_param = param_names[param_idx]
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
                        input_vector_for_actor = input_vector_for_actor.reshape(1, -1)
                        imputed_value = imputation_models[imputed_param][
                            imputed_actor
                        ].predict(input_vector_for_actor)
                        output_vector[actor_idx] = imputed_value
                # Assign imputed values to the dataset
                imputed_dataset[case_idx][param_idx][
                    :, :, i + lag_length
                ] = output_vector
        # For good measure, make sure there are no NaNs left in the dataset
        assert np.all(~np.isnan(imputed_dataset[case_idx]))
    if verbose:
        print("Imputation applied to HRV dataset")
    return imputed_dataset


if __name__ == "__main__":
    # Load dataset

    # Normalize data per case
    dataset = normalize_per_case(dataset)

    drop_first = True
    # Drop first and last phases from the data
    if drop_first:
        dataset = drop_edge_phases(dataset, drop_first=True, drop_last=False)
    # Impute data
    imputed_dataset = impute_data(dataset, lag_length=3)
    # Plot imputed data
