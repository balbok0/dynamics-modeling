from datetime import datetime
import os
from pathlib import Path
from turtle import hideturtle
from typing import List, Optional, Tuple
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np

from rosbag2torch import SequenceLookaheadDataset, filters, load_bags, readers
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import copy

from example_utils import StateControlBaseline, StateControlTrainableModel, augment_sequences_reflect_steer, unroll_sequence_torch, train, plot_rollout
from rosbag2torch.bag_processing.sequence_readers.abstract_sequence_reader import Sequences


def loss_from_batch(model: nn.Module, batch: Tuple[torch.Tensor, ...], criterion: nn.Module = nn.MSELoss()) -> torch.Tensor:
    # Unpack batch
    controls, states, targets, dts = batch

    # Convert to FloatTensor
    controls, states, targets, dts = controls.float(), states.float(), targets.float(), dts.float()

    # Forward pass - Unroll the trajectory
    predictions = unroll_sequence_torch(
        model=model,
        start_state=states[:, 0],
        controls=controls,
        dts=dts
    )

    # At each point of trajectory, calculate the loss
    rollout_losses = []
    for pred, target in zip(predictions, targets.transpose(0, 1)):
        loss = criterion(pred, target)
        rollout_losses.append(loss)

    # Total loss is the sum of the losses at each trajectory point
    loss = sum(rollout_losses)

    return loss


def single_hyperparameter_thread(
    train_sequences: Sequences,
    val_sequences: Sequences,
    features: List[str],
    delayed_features: List[str],
    delay_steps: float,
    rollout_s: float,
    hidden_size: float,
    num_hidden_layers: float,
    epochs: float,
    batch_size: float,
    activation_fn: nn.Module,
    lr: float,
    reader_str: str,
    log_hz: float,
    collapse_throttle_brake: bool,
    rollout_s_validation: float = 5.0,
):
    rollout_len = int(rollout_s  * log_hz)

    train_dataset = SequenceLookaheadDataset(train_sequences, features, delayed_features, delay_steps=delay_steps, sequence_length=rollout_len)
    val_dataset = SequenceLookaheadDataset(val_sequences, features, delayed_features, delay_steps=delay_steps, sequence_length=int(rollout_s_validation * log_hz))
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        return None

    settings_suffix = f"delay_{delay_steps}_rollout_{rollout_s}s_hidden_{hidden_size}_layers_{num_hidden_layers}_activation_{activation_fn.__class__.__name__}_lr_{lr:.2e}_bs_{batch_size}_epochs_{epochs}_reader_{reader_str}"
    date_suffix = datetime.now().strftime("%b%d_%H-%M-%S")

    model = StateControlTrainableModel(activation=activation_fn, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, collapse_throttle_brake=collapse_throttle_brake)
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01, lr=lr)
    criterion = nn.MSELoss()

    writer_name = f"runs/sequence_model_{settings_suffix}_{date_suffix}"
    model_prefix = f"models/sequence_model_{settings_suffix}_{date_suffix}"

    writer = SummaryWriter(log_dir=writer_name)

    val_loss = train(
        model=model,
        optimizer=optimizer,
        forward_fn=lambda *args, **kwargs: loss_from_batch(*args, **kwargs, criterion=criterion),
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        model_baseline=StateControlBaseline(delay_steps * 1.0 / log_hz, 0.001, 3.0),
        epochs=epochs,
        val_loader=DataLoader(val_dataset, batch_size=batch_size, shuffle=True),
        verbose=True,
        writer=writer
    )

    torch.save(model.state_dict(), f"{model_prefix}_state_dict.pt")
    torch.jit.script(model).save(f"{model_prefix}_scripted.pt")

    train_img = plot_rollout(model, train_dataset, rollout_s_validation)
    writer.add_image("Best Model (Train Sequence)", train_img, 0, dataformats="HWC")

    val_img = plot_rollout(model, val_dataset, rollout_s_validation)
    writer.add_image("Best Model (Val Sequence)", val_img, 0, dataformats="HWC")

    return writer_name, val_loss



def main():
    DATASET_TRAIN = "datasets/rzr_real"
    DATASET_VAL = "datasets/rzr_real_val"

    # What features to read.
    # NOTE: Values corresponding to these (and their ordering) are hardcoded in the model and train loop. Changing them here will break the code.
    features = ["control", "state"]
    delayed_features = ["target"]

    # Fixed Timestamp Reader
    log_hz = 30
    fixed_interval_reader = readers.FixedIntervalReader(
        list(set(features + delayed_features)),
        log_interval=1.0 / log_hz,
        filters=[
            filters.ForwardFilter(),
            filters.PIDInfoFilter()
        ]
    )
    # Async Reader
    async_reader = readers.ASyncSequenceReader(
        list(set(features + delayed_features)),
        features_to_record_on=["control"],
        filters=[
            filters.ForwardFilter(),
            filters.PIDInfoFilter()
        ]
    )


    # Train sequences
    fixed_interval_sequences = load_bags(DATASET_TRAIN, fixed_interval_reader)
    async_sequences = load_bags(DATASET_TRAIN, async_reader)

    # Augment train sequences
    fixed_interval_sequences = augment_sequences_reflect_steer(fixed_interval_sequences)
    async_sequences = augment_sequences_reflect_steer(async_sequences)

    # Validation Sequences
    val_sequences = load_bags(DATASET_VAL, async_reader)

    date_suffix = datetime.now().strftime("%b%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"runs/hyperparameter_search_sequence_model_{date_suffix}")

    NUM_HYPERPARAM_SEARCHES = 50
    for _ in range(NUM_HYPERPARAM_SEARCHES):
        # Sequence/Data Parameters
        delay_steps = np.random.randint(3, 8)  # indices
        # delay_steps = 7  # indices
        # rollout_s = 8  # seconds
        rollout_s = np.random.randint(1, 9)  # seconds

        # Model Parameters
        activation_fn = np.random.choice([nn.SELU(), nn.ReLU(), nn.LeakyReLU(), nn.Tanh()])
        hidden_size = int(2 ** np.random.randint(3, 8))
        num_hidden_layers = np.random.randint(1, 4)
        collapse_throttle_brake = bool(np.random.choice([True, False]))

        # Training Parameters
        epochs = 50  # Epochs are fixed
        lr = 10 ** np.random.uniform(-5, -2)
        batch_size = int(2 ** np.random.randint(6, 11))

        reader_str = np.random.choice(["fixed_interval", "async"])

        if reader_str == "fixed_interval":
            train_sequences = fixed_interval_sequences
        else:
            train_sequences = async_sequences

        hyperparam_settings = {
            "delay_steps": delay_steps,
            "rollout_s": rollout_s,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "batch_size": batch_size,
            "activation_fn": activation_fn,
            "lr": lr,
            "reader_str": reader_str,
            "collapse_throttle_brake": collapse_throttle_brake
        }

        hyperparam_writer_path, hyperparam_val_loss = single_hyperparameter_thread(
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            features=features,
            delayed_features=delayed_features,
            log_hz=log_hz,
            epochs=epochs,
            **hyperparam_settings
        )

        # For writing to tensorboard we want activation function to be a string
        hyperparam_settings["activation_fn"] = hyperparam_settings["activation_fn"].__class__.__name__

        writer.add_hparams(
            hparam_dict=hyperparam_settings,
            metric_dict={
                "val_loss": hyperparam_val_loss
            },
            run_name=hyperparam_writer_path
        )


if __name__ == "__main__":
    main()
