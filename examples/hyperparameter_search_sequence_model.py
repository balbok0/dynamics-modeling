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

from example_utils import reconstruct_poses_from_odoms, StateControlBaseline, StateControlTrainableModel, augment_sequences_reflect_steer
from rosbag2torch.bag_processing.sequence_readers.abstract_sequence_reader import Sequences


def get_world_frame_rollouts(model: nn.Module, states: torch.Tensor, controls: torch.Tensor, dts: torch.Tensor, rollout_in_seconds: float) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray]]:
    """Converts a sequence of robot frame states to world frame.
    It then uses a model to rollout the trajectory of length rollout_in_seconds in world frame, for each such interval in sequence.

    For example, if sequence if 10s long and rollout_in_seconds is 4s then this function will return:
        - Continuous sequence of true poses of length 10s
        - Start poses of each of predicted sequences (see below), corresponding to true poses at times 0s, 4s, 8s
        - Three continuos sequences of predicted poses of lengths 4s, 4s, 2s one for each interval in sequence,
            each starting at corresponding start pose.

    Args:
        model (nn.Module): Model to unroll the sequence with.
        states (torch.Tensor): (N, *) Tensor of robot frame states.
        controls (torch.Tensor): (N, *) Tensor of controls applied at each state.
        dts (torch.Tensor): (N, *) Tensor of time steps between each two consecutive states.
        rollout_in_seconds (float): Length of each rollout in seconds.

    Returns:
        Tuple[np.ndarray, np.ndarray, List[np.ndarray]]: Tuple of:
            - Continuous sequence of true poses for the whole sequence of length N + 1 (first pose will always be 0)
            - Start poses of each of predicted sequences (see below),
                corresponding to true poses at times 0s, rollout_in_seconds, 2*rollout_in_seconds, ...
            - List of predicted sequences, each of length rollout_in_seconds, starting at corresponding start pose.
    """
    controls, states, dts = controls.float(), states.float(), dts.float()

    # Controls need to be reshaped to also have batch size
    controls = controls.view(1, *controls.shape)
    dts = dts.view(1, *dts.shape)

    # Unroll the true trajectory
    poses_true = reconstruct_poses_from_odoms(states.numpy(), dts.numpy())

    # Get timestamp relative to the start of the rollout
    ts = np.cumsum(dts.numpy()) - dts.numpy()[0]

    # Iterate through the rollout in chunks of length ROLLOUT_S (in seconds)
    idx = 0  # index of the current rollout chunk
    start_poses = []  # poses at the start of each rollout chunk

    poses_pred_all = []

    while idx * rollout_in_seconds <= ts[-1]:
        # Get the current rollout chunk
        cur_idx = np.where(ts // rollout_in_seconds == idx)[0]

        # Get the corresponding dts, controls and states
        cur_dts = dts[:, cur_idx]
        cur_controls = controls[:, cur_idx]
        cur_states = states[cur_idx]

        # Unroll the model predictions for the current rollout chunk
        predictions = unroll_sequence_torch(
            model=model,
            start_state=cur_states[None, 0],
            controls=cur_controls,
            dts=cur_dts
        )

        # Convert to numpy
        np_predictions = np.array(
            [pred.detach().cpu().numpy() for pred in predictions]
        ).squeeze()

        # Convert (dx, dtheta) to (x, y, theta)
        poses_pred = reconstruct_poses_from_odoms(np_predictions, cur_dts.numpy(), start_pose=poses_true[cur_idx[0]])

        # Save the start pose for this chunk
        start_poses.append(poses_true[cur_idx[0]])

        poses_pred_all.append(poses_pred)
        idx += 1

    return poses_true, np.array(start_poses), poses_pred_all


def unroll_sequence_torch(
    model: nn.Module,
    start_state: torch.Tensor,
    controls: torch.Tensor,
    dts: torch.Tensor,
) -> List[torch.Tensor]:
    """ Unroll the model forward for a sequence of states.
    In this function both states in and out are in robot frame.

    Args:
        model (nn.Module): Model to use for prediction of each step.
        start_state (torch.Tensor): State to start rollout from.
            Shape: (N, *), where * is the state shape
        controls (torch.Tensor): Controls to be applied at each step of the rollout.
            (N, S, *), where N is the batch size, S is the sequence length, and * is the control shape.
        dts (torch.Tensor): Difference in time between each step of the rollout.
            (N, S), where N is the batch size, and S is the sequence length

    Returns:
        List[torch.Tensor]: A List of length S, where each element is a (N, *) Tensor, where * is the state shape
    """
    controls = controls.transpose(0, 1)
    dts = dts.transpose(0, 1)

    cur_state = start_state
    result = []
    for control, dt in zip(controls, dts):
        # acceleration * dt + prev state
        # m / s^2 * s + m / s
        cur_state = model(control=control, state=cur_state) * dt.view(-1, 1) + cur_state

        result.append(cur_state)
    return result


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    epochs: int,
    val_loader: Optional[DataLoader] = None,
    model_baseline: Optional[nn.Module] = None,
    verbose: bool = True,
    writer: Optional[SummaryWriter] = None,
):
    best_val_loss = np.inf
    best_state_dict = None

    trange_epochs = trange(epochs, desc="Epochs", disable=not verbose, leave=True)
    for epoch in trange_epochs:
        running_loss_rollout_steps = defaultdict(float)
        running_total_loss = 0.0
        running_baseline_loss = 0.0
        for controls, states, targets, dts in tqdm(train_loader, disable=not verbose, desc="Train", leave=False):
            # Convert to FloatTensor
            controls, states, targets, dts = controls.float(), states.float(), targets.float(), dts.float()

            # Zero-out the gradient
            optimizer.zero_grad()

            # Forward pass - Unroll the trajectory
            predictions = unroll_sequence_torch(
                model=model,
                start_state=states[:, 0],
                controls=controls,
                dts=dts
            )


            # At each point of trajectory, calculate the loss
            rollout_losses = []
            for rollout_idx, (pred, target) in enumerate(zip(predictions, targets.transpose(0, 1))):
                loss = criterion(pred, target)
                rollout_losses.append(loss)
                running_loss_rollout_steps[rollout_idx] += loss.detach().cpu().item()

            # Total loss is the sum of the losses at each trajectory point
            loss = sum(rollout_losses)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Log loss
            running_total_loss += loss.detach().cpu().item()

            # Repeat for baseline model
            with torch.no_grad():
                if model_baseline is not None:
                    baseline_predictions = unroll_sequence_torch(
                        model=model_baseline,
                        start_state=states[:, 0],
                        controls=controls,
                        dts=dts
                    )
                    baseline_rollout_losses = []
                    for rollout_idx, (pred, target) in enumerate(zip(baseline_predictions, targets.transpose(0, 1))):
                        loss = criterion(pred, target)
                        baseline_rollout_losses.append(loss)
                        # running_baseline_loss_rollout_steps[rollout_idx] += loss.detach().cpu().item()

                    # Total loss is the sum of the losses at each trajectory point
                    baseline_loss = sum(baseline_rollout_losses)
                    running_baseline_loss += baseline_loss.detach().cpu().item()

        # Run zero_grad at the end of each epoch, just in case
        optimizer.zero_grad()

        train_loss = running_total_loss / len(train_loader)
        train_baseline_loss = running_baseline_loss / len(train_loader)
        if writer is not None:
            writer.add_scalar("Loss/train total", train_loss, epoch)
            if model_baseline is not None:
                writer.add_scalar("Loss/train total baseline", train_baseline_loss, epoch)
            for rollout_idx, rollout_loss in running_loss_rollout_steps.items():
                writer.add_scalar(f"Loss/train @ rollout step {rollout_idx}", rollout_loss / len(train_loader), epoch)
        desc = f"Epochs Train Loss {train_loss:.4g} Baseline {train_baseline_loss:.4g}"

        if val_loader is not None:
            with torch.no_grad():
                running_loss = 0.0
                running_loss_rollout_steps = defaultdict(float)

                running_baseline_loss = 0.0

                for controls, states, targets, dts in tqdm(val_loader, disable=not verbose, desc="Val", leave=False):
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
                    for rollout_idx, (pred, target) in enumerate(zip(predictions, targets.transpose(0, 1))):
                        loss = criterion(pred, target)
                        rollout_losses.append(loss)
                        running_loss_rollout_steps[rollout_idx] += loss.detach().cpu().item()

                    loss = sum(rollout_losses)
                    running_loss += loss.detach().cpu().item()

                    # Repeat for baseline model
                    if model_baseline is not None:
                        baseline_predictions = unroll_sequence_torch(
                            model=model_baseline,
                            start_state=states[:, 0],
                            controls=controls,
                            dts=dts
                        )
                        baseline_rollout_losses = []
                        for rollout_idx, (pred, target) in enumerate(zip(baseline_predictions, targets.transpose(0, 1))):
                            loss = criterion(pred, target)
                            baseline_rollout_losses.append(loss)
                            # running_baseline_loss_rollout_steps[rollout_idx] += loss.detach().cpu().item()

                        # Total loss is the sum of the losses at each trajectory point
                        baseline_loss = sum(baseline_rollout_losses)
                        running_baseline_loss += baseline_loss.detach().cpu().item()


                val_loss = running_loss / len(val_loader)
                val_baseline_loss = running_baseline_loss / len(val_loader)
                if writer is not None:
                    writer.add_scalar("Loss/val total", val_loss, epoch)
                    if model_baseline is not None:
                        writer.add_scalar("Loss/val total baseline", val_baseline_loss, epoch)
                    for rollout_idx, rollout_loss in running_loss_rollout_steps.items():
                        writer.add_scalar(f"Loss/val @ rollout step {rollout_idx}", rollout_loss / len(val_loader), epoch)
                desc += f" Val Loss {val_loss:.4g} Baseline {val_baseline_loss:.4g}"

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state_dict = copy.deepcopy(model.state_dict())

        trange_epochs.set_description(desc)

    # Load best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return best_val_loss


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
        train_loader=DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        model_baseline=StateControlBaseline(delay_steps * 1.0 / log_hz, 0.001, 3.0),
        criterion=criterion,
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


def plot_rollout(model, dataset, rollout_s):
    fig, ax = plt.figure(figsize=(10, 10)), plt.subplot(111)
    ax: plt.Axes

    controls, states, targets, dts = dataset.longest_rollout
    poses_true, start_poses, poses_pred_all = get_world_frame_rollouts(model, states, controls, dts, rollout_in_seconds=rollout_s)

    ax.scatter(start_poses[:, 0], start_poses[:, 1], color="red", marker="X")
    ax.plot(poses_true[:, 0], poses_true[:, 1], color="red", label="True")
    for idx, poses_pred in enumerate(poses_pred_all):
        plt_kwargs = {"color": "black"}
        if idx == 0:
            plt_kwargs["label"] = "Predicted"
        ax.plot(poses_pred[:, 0], poses_pred[:, 1], **plt_kwargs)

    ax.legend()
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")

    #Image from plot
    ax.axis('off')
    fig.tight_layout(pad=0)

    # To remove the huge white borders
    ax.margins(0)

    fig.canvas.draw()
    image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    plt.close(fig)

    return image_from_plot


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