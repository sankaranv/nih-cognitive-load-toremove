from mcmc_impute import *
from utils.data import *
from models.nn import *
from models.networks import *
import argparse
import pickle
import torch
import torch.nn as nn
import wandb

if __name__ == "__main__":
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    dataset = pickle.load(open("./data/imputed_dataset.pkl", "rb"))
    param_names = ["PNS index", "SNS index", "Mean RR", "RMSSD", "LF-HF"]
    role_names = ["Anes", "Nurs", "Perf", "Surg"]
    num_actors = len(role_names)
    seq_len = 10
    pred_len = 5
    num_features = 11
    hrv_dataset = HRVDataset(dataset, seq_len, pred_len, param_names, param_names[0])
    train_loader, val_loader, test_loader = create_torch_dataloaders(
        hrv_dataset, batch_size=64, train_ratio=0.8, val_ratio=0.1
    )

    # Set up model
    model = OneLayerLinear(seq_len, pred_len, num_actors, num_features)
    model.to(device)

    # Hyperparameters
    lr = 0.001
    num_epochs = 100
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    wandb.config.update(
        {
            "lr": lr,
            "num_epochs": num_epochs,
            "criterion": criterion,
            "optimizer": optimizer,
        }
    )

    # Train model
    wandb.init(project="nih-cognitive-load")
    wandb.watch(model)
    for epoch in range(num_epochs):
        train_loss = model.train(train_loader, optimizer, criterion, device)
        val_loss = model.val(val_loader, criterion, device)
        print(f"Epoch {epoch + 1}: Train Loss: {train_loss}, Val Loss: {val_loss}")

    # Test model
    test_loss = model.test(test_loader)
    print(f"Test Loss: {test_loss}")
