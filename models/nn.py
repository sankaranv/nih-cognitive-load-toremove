import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# from utils.data import *
import wandb
import pickle


def create_input_output_pairs(dataset, seq_len, pred_len, param_names):
    input_output_pairs = {}
    for key in param_names:
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
            # Use the mask to check if there are any missing values in the last seq_len + pred_len timesteps
            # If there are no missing values, add to training set
            # The first seq_len steps are used as input and the next pred_len are used as output
            for i in range(num_timesteps - seq_len - pred_len):
                if np.all(mask[i : i + seq_len + pred_len]):
                    key = param_names[param_idx]
                    input_vector = param_data[:, :, i : i + seq_len]
                    output_vector = param_data[
                        :, :, i + seq_len : i + seq_len + pred_len
                    ]
                    assert np.all(~np.isnan(output_vector))
                    input_output_pairs[key].append(
                        (
                            torch.FloatTensor(input_vector),
                            torch.FloatTensor(output_vector),
                        )
                    )
    return input_output_pairs


class HRVDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, seq_len, pred_len, param_names, param):
        self.dataset = dataset
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.param_names = param_names
        all_input_output_pairs = create_input_output_pairs(
            dataset, seq_len, pred_len, param_names
        )
        self.input_output_pairs = all_input_output_pairs[param]

    def __len__(self):
        return len(self.input_output_pairs)

    def __getitem__(self, idx):
        X = self.input_output_pairs[idx][0]
        y = self.input_output_pairs[idx][1][:, 0, :]
        return X, y


def create_torch_dataloaders(hrv_dataset, batch_size, train_ratio, val_ratio):
    # Split into train, val, test sets
    train_size = int(train_ratio * len(hrv_dataset))
    val_size = int(val_ratio * len(hrv_dataset))
    test_size = len(hrv_dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        hrv_dataset, [train_size, val_size, test_size]
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True
    )
    return train_loader, val_loader, test_loader


class NNModel(nn.Module):
    def __init__(self, seq_len, pred_len, num_actors, num_features):
        super().__init__()
        self.model = self.create_model(seq_len, pred_len, num_actors, num_features)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_features = num_features

    def create_model(self):
        pass

    def forward(self, x):
        # Input dim is (batch_size, num_actors, num_features, seq_len)
        # Reshape to (batch_size, seq_len * num_actors * num_features)
        # Output dim is (batch_size, num_actors, pred_len) since we only predict HRV value
        x = x.view(x.size(0), -1)
        out = self.model(x)
        out = out.view(out.size(0), -1, self.pred_len)
        return out

    def train(self, train_loader, optimizer, criterion, device):
        self.model.train()
        train_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = self.forward(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            wandb.log({"Train Loss": loss.item()})
            self.val(train_loader, criterion, device)
        return train_loss / len(train_loader.dataset)

    def val(self, train_loader, criterion, device):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                output = self.forward(data)
                val_loss += criterion(output, target).item()
        val_loss /= len(train_loader.dataset)
        wandb.log({"Val Loss": val_loss})
        return val_loss

    def test(self, test_loader, criterion, device):
        self.model.eval()
        test_loss = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.forward(data)
                test_loss += criterion(output, target).item()
        test_loss /= len(test_loader.dataset)
        wandb.log({"Test Loss": test_loss})
        return test_loss
