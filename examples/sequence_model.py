from datetime import datetime
from typing import List, Optional
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np

from rosbag2torch import SequenceLookaheadDataset, filters, load_bags, readers
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from example_utils import reconstruct_from_odoms


class Model(nn.Module):
    def __init__(self, activation: nn.Module, hidden_size: int, num_hidden_layers: int) -> None:
        super().__init__()
        assert hidden_size > 0
        assert num_hidden_layers >= 0

        # Edge case: no hidden layers
        if num_hidden_layers == 0:
            self._model = nn.Linear(5, 2)
        else:
            self._model = nn.Sequential(
                nn.Linear(5, hidden_size),
                nn.SELU(),
                *[
                    nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.SELU())
                    for _ in range(num_hidden_layers - 1)
                ],
                nn.Linear(hidden_size, 2)
            )

    def forward(self, control: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self._model(torch.cat((control, state), dim=1))


class ModelBaseline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, control: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return state


def unroll_sequence_torch(
    model: nn.Module,
    start_state: torch.Tensor,
    controls: torch.Tensor,
    dts: torch.Tensor,
) -> List[torch.Tensor]:
    """

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
        cur_state = model(control, cur_state) * dt.view(-1, 1) + cur_state

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
                    for rollout_idx, (pred, target) in enumerate(zip(predictions, targets.transpose(0, 1))):
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
            writer.add_scalar("Loss/train", train_loss, epoch)
            if model_baseline is not None:
                writer.add_scalar("Loss/train baseline", train_baseline_loss, epoch)
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
                        for rollout_idx, (pred, target) in enumerate(zip(predictions, targets.transpose(0, 1))):
                            loss = criterion(pred, target)
                            baseline_rollout_losses.append(loss)
                            # running_baseline_loss_rollout_steps[rollout_idx] += loss.detach().cpu().item()

                        # Total loss is the sum of the losses at each trajectory point
                        baseline_loss = sum(baseline_rollout_losses)
                        running_baseline_loss += baseline_loss.detach().cpu().item()


                val_loss = running_loss / len(val_loader)
                val_baseline_loss = running_baseline_loss / len(val_loader)
                if writer is not None:
                    writer.add_scalar("Loss/val", val_loss, epoch)
                    if model_baseline is not None:
                        writer.add_scalar("Loss/val baseline", val_baseline_loss, epoch)
                    for rollout_idx, rollout_loss in running_loss_rollout_steps.items():
                        writer.add_scalar(f"Loss/val @ rollout step {rollout_idx}", rollout_loss / len(val_loader), epoch)
                desc += f" Val Loss {val_loss:.4g} Baseline {val_baseline_loss:.4g}"

        trange_epochs.set_description(desc)


def main():
    # "What to do" Parameters
    TRAIN = True
    PLOT_VAL = True

    # Sequence/Data Parameters
    DELAY_STEPS = 3  # indices
    ROLLOUT_S = 5  # seconds

    # Model Parameters
    ACTIVATION_FN = nn.ReLU()
    HIDDEN_SIZE = 64
    NUM_HIDDEN_LAYERS = 2

    # Training Parameters
    EPOCHS = 50
    LR = 1e-3
    BATCH_SIZE = 32

    # Suffix to use for saving this configuration. Shared for tensorboard and models
    settings_suffix = f"delay_{DELAY_STEPS}_rollout_{ROLLOUT_S}s_hidden_{HIDDEN_SIZE}_layers_{NUM_HIDDEN_LAYERS}_activation_{ACTIVATION_FN.__class__.__name__}_lr_{LR:.2e}_bs_{BATCH_SIZE}_epochs_{EPOCHS}"

    # What features to read.
    # NOTE: Values corresponding to these (and their ordering) are hardcoded in the model and train loop. Changing them here will break the code.
    features = ["control", "state"]
    delayed_features = ["target"]

    # # Async Reader
    # reader = readers.ASyncSequenceReader(
    #     list(set(features + delayed_features)),
    #     features_to_record_on=["control"],
    #     filters=[
    #         filters.ForwardFilter(),
    #         filters.PIDInfoFilter()
    #     ]
    # )

    # Fixed Timestamp Reader
    log_hz = 30
    reader = readers.FixedIntervalReader(
        list(set(features + delayed_features)),
        log_interval=1.0 / log_hz,
        filters=[
            filters.ForwardFilter(),
            filters.PIDInfoFilter()
        ]
    )
    rollout_len = int(ROLLOUT_S  * log_hz)

    val_sequences = load_bags("datasets/rzr_real_val", reader)
    val_dataset = SequenceLookaheadDataset(val_sequences, features, delayed_features, delay_steps=DELAY_STEPS, sequence_length=rollout_len)
    model_prefix = f"models/sequence_model_{settings_suffix}"


    if TRAIN:
        model = Model(activation=ACTIVATION_FN, hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_HIDDEN_LAYERS)
        optimizer = optim.Adam(model.parameters(), weight_decay=0.01, lr=LR)
        criterion = nn.MSELoss()

        train_sequences = load_bags("datasets/rzr_real", reader)
        train_dataset = SequenceLookaheadDataset(
            train_sequences, features, delayed_features, delay_steps=DELAY_STEPS, sequence_length=rollout_len
        )

        writer = SummaryWriter(log_dir=f"runs/sequence_model_{settings_suffix}_{datetime.now().strftime('%b%d_%H-%M-%S')}")

        train(
            model=model,
            optimizer=optimizer,
            train_loader=DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True),
            criterion=criterion,
            epochs=EPOCHS,
            val_loader=DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True),
            verbose=True,
            writer=writer
        )

        torch.save(model.state_dict(), f"{model_prefix}_state_dict.pt")
        torch.jit.script(model).save(f"{model_prefix}_scripted.pt")

    if PLOT_VAL:
        with torch.no_grad():
            if "model" not in vars():
                model = Model(activation=ACTIVATION_FN, hidden_size=HIDDEN_SIZE, num_hidden_layers=NUM_HIDDEN_LAYERS)
                model.load_state_dict(torch.load(f"{model_prefix}_state_dict.pt"))
            model.eval()

            controls, states, targets, dts = val_dataset.longest_rollout
            controls, states, targets, dts = controls.float(), states.float(), targets.float(), dts.float()

            # Controls need to be reshaped to also have batch size
            controls = controls.view(1, *controls.shape)
            dts = dts.view(1, *dts.shape)

            # Unroll the true trajectory
            poses_true = reconstruct_from_odoms(states.numpy(), dts.numpy())

            # Get timestamp relative to the start of the rollout
            ts = np.cumsum(dts.numpy()) - dts.numpy()[0]

            # Iterate through the rollout in chunks of length ROLLOUT_S (in seconds)
            idx = 0  # index of the current rollout chunk
            start_poses = []  # poses at the start of each rollout chunk

            while idx * ROLLOUT_S <= ts[-1]:
                # Get the current rollout chunk
                cur_idx = np.where(ts // ROLLOUT_S == idx)[0]

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
                poses_pred = reconstruct_from_odoms(np_predictions, cur_dts.numpy(), start_pose=poses_true[cur_idx[0]])
                poses_pred = np.vstack((poses_true[None, cur_idx[0]], poses_pred))  # Add start pose to the beginning of the sequence

                # Save the start pose for this chunk
                start_poses.append(poses_true[cur_idx[0]])

                # Plot the (x, y) poses
                plt_kwargs = {"color": "black"}
                if idx == 0:
                    plt_kwargs["label"] = "Predicted"
                plt.plot(poses_pred[:, 0], poses_pred[:, 1], **plt_kwargs)

                idx += 1

            start_poses = np.array(start_poses)
            plt.scatter(start_poses[:, 0], start_poses[:, 1], color="red", marker="X")

            plt.plot(poses_true[:, 0], poses_true[:, 1], color="red", label="True")
            plt.legend()
            plt.xlabel("x")
            plt.ylabel("y")
            plt.savefig(f"plots/sequence_model_val_{settings_suffix}.png", bbox_inches="tight")
            plt.show()

if __name__ == "__main__":
    main()
