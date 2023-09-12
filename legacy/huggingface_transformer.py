from models.transformer import ContinuousTransformer
from utils.legacy.data_hf import *
import argparse
import torch
import sys
import wandb
from transformers import (
    TimeSeriesTransformerForPrediction,
    TimeSeriesTransformerConfig,
)


if __name__ == "__main__":
    # Experiment settings and hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--time_interval", type=int, default=5)
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
    parser.add_argument("--input_window", type=int, default=122 - 32)
    parser.add_argument("--output_window", type=int, default=92)
    args = parser.parse_args()

    # Set up WandB for logging
    # wandb.init(project="nih-cognitive-load", config=args)

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

    # Load dataset
    # dataset = make_dataset_from_file(
    #     data_dir="./data", time_interval=args.time_interval, param_id=hrv_param_idx
    # )
    # max_len = get_max_len(dataset)
    dataset, lengths = make_dataset_from_file(
        data_dir="./data", time_interval=args.time_interval, param_id=hrv_param_idx
    )
    temporal_features = make_temporal_features(dataset, lengths)
    # Create train-val-test split
    (
        train_dataset,
        val_dataset,
        test_dataset,
        train_temporal_features,
        val_temporal_features,
        test_temporal_features,
    ) = make_train_test_split(
        dataset, temporal_features, args.train_split, args.val_split, args.test_split
    )

    # Create HRVDataset objects
    train_dataset = HRVDataset(train_dataset, train_temporal_features)
    val_dataset = HRVDataset(val_dataset, val_temporal_features)
    test_dataset = HRVDataset(test_dataset, test_temporal_features)

    # Create input-output sequences for training
    in_out_seq_data, in_out_seq_temporal = get_input_output_sequences(
        args.input_window, args.output_window, train_dataset
    )

    # Model Config
    config = TimeSeriesTransformerConfig(
        prediction_length=32,
        context_length=64,
        distribution_output="normal",
        input_size=4,
        scaling="mean",
        num_time_features=11,
        num_dynamic_real_features=4,
        num_static_categorical_features=0,
        num_static_real_features=0,
        cardinality=None,
        embedding_dimension=None,
        d_model=64,
        encoder_layers=2,
        decoder_layers=2,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=64,
        decoder_ffn_dim=64,
        activation_function="gelu",
        dropout=0.1,
        encoder_layerdrop=0.1,
        decoder_layerdrop=0.1,
        attention_dropout=0.1,
        activation_dropout=0.1,
        num_parallel_samples=100,
        init_std=0.2,
        use_cache=True,
    )

    # Create model
    model = TimeSeriesTransformerForPrediction(config=config)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)
    model.train()

    for epoch in range(args.epochs):
        for batch, start_idx in enumerate(
            range(0, len(train_dataset), args.batch_size)
        ):
            # Get a batch of sequences and push them to GPU if available
            input_seq, target_seq, input_temporal, output_temporal = get_batch(
                in_out_seq_data, in_out_seq_temporal, args.batch_size, start_idx
            )

            input_seq = input_seq.to(device)
            target_seq = target_seq.to(device)
            input_temporal = input_temporal.to(device)
            output_temporal = output_temporal.to(device)

            print(
                input_seq.shape,
                target_seq.shape,
                input_temporal.shape,
                output_temporal.shape,
            )

            # Get mask for positions where there is NaN in the data
            input_mask = get_mask(input_seq).to(device)
            output_mask = get_mask(target_seq).to(device)

            # Forward pass
            optimizer.zero_grad()
            output = model(
                past_values=input_seq,  # Should be shape (batch size, seq len, n_features)
                past_time_features=input_temporal,  # Should be shape (batch size, seq len, n_time_features)
                past_observed_mask=input_mask,
                future_values=target_seq,
                future_observed_mask=output_mask,
                future_time_features=output_temporal,
                return_dict=True,
            )

            # Backward pass
            loss = criterion(output, target_seq)
            loss.backward()

            # Clip gradients to prevent exploding gradients and then update parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Log results on WandB and stdout
            if batch % args.log_freq == 0:
                # wandb.log({"loss": loss.item()})
                print(f"Epoch: {epoch}, Batch: {batch}, Loss: {loss.item()}")

            # Save model checkpoint
            if batch % args.save_freq == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(
                        args.save_dir,
                        f"transformer/transformer_{args.time_interval}_{args.hrv_param}_{epoch}_{batch}.pt",
                    ),
                )
    # Save trained model
    torch.save(
        model.state_dict(),
        os.path.join(
            args.save_dir,
            f"transformer_{args.time_interval}_{args.hrv_param}.pt",
        ),
    )

# wandb.finish()
