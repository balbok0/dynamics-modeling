from matplotlib import pyplot as plt
from rosbag2torch import load_bags, readers, filters, LookaheadSequenceDataset
from typing import Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import numpy as np


def reconstruct_from_odoms(d_odom: np.ndarray, dt: np.ndarray, start_pose: Optional[np.ndarray] = None, delay_steps: int = 1):
    """This function reconstructs the trajectory from odometry data.
    Odometry data is assumed to be in the form of [v_x, v_y, v_theta] or [v_x, v_theta],
    with dx and dy being in the robot frame.

    Args:
        d_odom (np.ndarray): A (n, 2) or (n, 3) array of odometry data.
        dt (np.ndarray): A (n,) array of differences in timestamps.
            It needs take delay_steps into account.
            This means that: dt[i] = t[i + delay_steps] - t[i]
        start_pose (Optional[np.ndarray], optional): A (3,) array of the starting pose (x, y, theta).
            Defaults to (0, 0, 0).
        delay_steps (int, optional): Number of steps taken for each prediction. Defaults to 1.

    Returns:
        _type_: _description_
    """
    assert len(d_odom.shape) == 2 and d_odom.shape[1] in {2, 3}, "d_odom must be a 2D array with 2 (dx, dtheta) or 3 (dx, dy, dtheta) columns"
    assert delay_steps >= 1, "Delay steps must be at least 1"

    # We expect dt to be a 1D array. It may be a (n, 1) array, in which case we'll reshape it to (n,).
    dt = dt.squeeze()

    d_x = d_odom[:, 0]
    if d_odom.shape[1] == 2:
        d_y = np.zeros_like(d_x)
        d_theta = d_odom[:, 1]
    else:
        # Shape == 3
        d_y = d_odom[:, 1]
        d_theta = d_odom[:, 2]

    if start_pose is None:
        start_pose = np.array([0.0, 0.0, 0.0])

    # Rollout each continuous sequence seperately
    # Continuos sequence is meant by markovian chain where theta(i) = d_theta(i, j) + theta(j)
    size_rollout = len(d_x) // delay_steps
    delayed_size  = size_rollout * delay_steps
    delayed_shape = (size_rollout, delay_steps)

    # Reshape time series to (size_rollout, delay_steps)
    t_delayed = dt[:delayed_size].reshape(delayed_shape)

    thetas = np.reshape(d_theta[:delayed_size], delayed_shape)
    thetas = np.cumsum(thetas * t_delayed, axis=0) + start_pose[2]
    plt.plot(np.cumsum(dt[:len(thetas)]), thetas)
    plt.show()

    # Create vectors along and orthogonal to theta
    along_vec = np.concatenate((np.cos(thetas)[:, None], np.sin(thetas)[:, None]), axis=2)
    # Orthogonal vector is -sin(theta) along x and cos(theta) along y, so we can just use along
    ortho_vec = along_vec[..., [1, 0]]
    ortho_vec[..., 0] *= -1

    along_vals = np.reshape(d_x[:delayed_size], delayed_shape)
    ortho_vals = np.reshape(d_y[:delayed_size], delayed_shape)

    poses = np.cumsum(t_delayed[:, None] * (along_vec * along_vals[:, None] + ortho_vec * ortho_vals[:, None]), axis=0)

    poses = np.transpose(poses, (1, 0, 2)).reshape(-1, 2)
    poses += start_pose[:2]

    return poses


def train(
    model: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    criterion: nn.modules.loss._Loss,
    epochs: int,
    val_loader: Optional[DataLoader] = None,
    verbose: bool = True,
):
    writer = SummaryWriter()

    trange_epochs = trange(epochs, desc="Epochs", disable=not verbose, leave=True)
    for epoch in trange_epochs:
        running_loss = 0.0
        for x, y, dt in tqdm(train_loader, disable=not verbose, desc="Train", leave=False):
            optimizer.zero_grad()
            # acceleration * dt + prev state
            # m / s^2 * s + m / s
            y_pred = model(x) * dt

            loss = criterion(y_pred, y - x[:, -2:])
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().item()

        optimizer.zero_grad()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        desc = f"Epochs Train Loss {train_loss:.4g}"

        if val_loader is not None:
            val_running_loss = 0.0
            for x, y, dt in tqdm(val_loader, disable=not verbose, desc="Validation", leave=False):
                optimizer.zero_grad()
                y_pred = model(x) * dt

                loss = criterion(y_pred, y - x[:, -2:])
                loss.backward()
                optimizer.step()

                val_running_loss += loss.detach().cpu().item()

            val_loss = val_running_loss / len(val_loader)
            writer.add_scalar("Loss/val", val_loss, epoch)
            desc += f" Val Loss {val_loss:.4g}"

        trange_epochs.set_description(desc)


def main():
    DELAY_STEPS = 1
    EPOCHS = 500
    TRAIN = False
    PLOT_VAL = False

    reader = readers.ASyncSequenceReader(
        ["control", "state", "target"],
        features_to_record_on=["control"],
        filters=[
            filters.ForwardFilter(),
            filters.PIDInfoFilter()
        ]
    )

    val_sequences = load_bags("datasets/rzr_real_val", reader)


    if TRAIN:
        model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
        criterion = nn.MSELoss()

        train_sequences = load_bags("datasets/rzr_real", reader)
        train_dataset = LookaheadSequenceDataset(train_sequences, ["control", "state"], ["target"], delay_steps=DELAY_STEPS)

        val_dataset = LookaheadSequenceDataset(val_sequences, ["control", "state"], ["target"], delay_steps=DELAY_STEPS)

        train(
            model,
            optimizer,
            DataLoader(train_dataset, batch_size=32, shuffle=True),
            criterion,
            epochs=EPOCHS,
            val_loader=DataLoader(val_dataset, batch_size=32, shuffle=True),
            verbose=True,
        )

        torch.jit.script(model).save("models/example_hidden2relu_model.pt")
    else:
        model = torch.jit.load("models/example_hidden2relu_model.pt")

    if PLOT_VAL:
        # Evaluate on the validation set
        model.eval()
        with torch.no_grad():
            # Take the longest sequence from validation set
            longest_val_sequence = val_sequences[np.argmax(np.array([len(s[list(s.keys())[0]]) for s in val_sequences]))]

            y_true_all = []
            y_zero_all = []
            y_pred_all = []
            dts_all = []

            for x, y, dt in tqdm(DataLoader(LookaheadSequenceDataset([longest_val_sequence], ["control", "state"], ["target"], delay_steps=DELAY_STEPS), batch_size=1, shuffle=False), desc="Final Validation"):
                y_pred = model(x) * dt + x[:, -2:]
                y_zero_all.extend(x[:, -2:].detach().cpu().numpy())
                y_pred_all.extend(y_pred.detach().cpu().numpy())
                y_true_all.extend(y.detach().cpu().numpy())
                dts_all.extend(dt.detach().cpu().numpy())

            y_true_all = np.array(y_true_all)
            tmp = np.zeros((len(y_true_all), 3))
            tmp[:, 0] = y_true_all[:, 0]
            tmp[:, 2] = y_true_all[:, 1]
            y_true_all = tmp

            y_zero_all = np.array(y_zero_all)
            tmp = np.zeros((len(y_zero_all), 3))
            tmp[:, 0] = y_zero_all[:, 0]
            tmp[:, 2] = y_zero_all[:, 1]
            y_zero_all = tmp

            y_pred_all = np.array(y_pred_all)
            tmp = np.zeros((len(y_pred_all), 3))
            tmp[:, 0] = y_pred_all[:, 0]
            tmp[:, 2] = y_pred_all[:, 1]
            y_pred_all = tmp

            dts_all = np.array(dts_all)

            # Plot the dx, dtheta for each
            fig, axs = plt.subplots(3, 2, figsize=(10, 10))
            for row_idx, (y, type_y) in enumerate(zip([y_true_all, y_zero_all, y_pred_all], ["True", "Zero", "Pred"])):
                # for col_idx, col_name in enumerate(["dx", "dtheta"]):
                #     axs[row_idx, col_idx].plot(np.cumsum(dts_all), y[:, col_idx])
                #     axs[row_idx, col_idx].set_title(f"{type_y} - {col_name}")
                for col_idx, (y_idx, col_name) in enumerate(zip([0, 2], ["dx", "dtheta"])):
                    axs[row_idx, col_idx].plot(np.cumsum(dts_all), y[:, y_idx])
                    axs[row_idx, col_idx].set_title(f"{type_y} - {col_name}")
            plt.show()

            poses_true = reconstruct_from_odoms(y_true_all, dts_all, delay_steps=DELAY_STEPS)
            poses_zero = reconstruct_from_odoms(y_zero_all, dts_all, delay_steps=DELAY_STEPS)
            poses_pred = reconstruct_from_odoms(y_pred_all, dts_all, delay_steps=DELAY_STEPS)

            plt.plot(poses_true[:, 0], poses_true[:, 1], label="True")
            plt.plot(poses_zero[:, 0], poses_zero[:, 1], label="Zero")
            plt.plot(poses_pred[:, 0], poses_pred[:, 1], label="Pred")
            plt.legend()
            plt.show()


if __name__ == "__main__":
    main()
