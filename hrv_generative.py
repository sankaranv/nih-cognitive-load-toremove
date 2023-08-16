from models.transformer import ContinuousTransformer
from utils.data import *
import argparse
import torch
import wandb

if __name__ == "__main__":
    # Experiment settings and hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_interval", type=str, default="5min")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--d_hidden", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--save_freq", type=int, default=10)
    parser.add_argument("--log_freq", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--cuda", type=bool, default=False)
    parser.add_argument("--hrv_param", type=str, default="Mean RR")
    parser.add_argument("--train_split", type=float, default=0.6)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--test_split", type=float, default=0.2)
    args = parser.parse_args()

    # Set up WandB for logging
    wandb.init(project="nih-cognitive-load", config=args)

    # Set random seed
    torch.manual_seed(args.seed)

    # Set device
    device = torch.device("cuda" if args.cuda else "cpu")

    # Set HRV parameter to model
    if args.hrv_param not in param_names:
        raise ValueError(
            f"Invalid HRV parameter. Available parameters are: {param_names}"
        )
    hrv_param_idx = param_indices[args.hrv_param]

    # Load dataset and create train-val-test split
    dataset = make_dataset_from_file(args.time_interval, hrv_param_idx)
    train_dataset, val_dataset, test_dataset = make_train_test_split(
        dataset, args.train_split, args.val_split, args.test_split
    )
    train_dataset = HRVDataset(train_dataset)
    val_dataset = HRVDataset(val_dataset)
    test_dataset = HRVDataset(test_dataset)

    # Create input-output sequences for training
    train_inputs, train_outputs = get_input_output_sequences(train_dataset)

    # Set up loss and optimizer for training
    model = ContinuousTransformer()
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

    # Start training
    model.train()

    for epoch in range(args.epochs):
        for batch, start_idx in enumerate(
            range(0, len(train_dataset), args.batch_size)
        ):
            # Get a batch of sequences
            input_seq, target_seq = get_batch(
                train_inputs, train_outputs, start_idx, args.batch_size
            )

            optimizer.zero_grad()
            output = model(input_seq)
            loss = criterion(output, target_seq)
            loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Log results on WandB
            if batch % args.log_freq == 0:
                wandb.log({"loss": loss.item()})


wandb.finish()
