from turtle import forward
from typing import Optional
from joblib import delayed

from rosbag2torch import SequenceLookaheadDataset, filters, load_bags, readers
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.__model = nn.Sequential(
            nn.Linear(5, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, control: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return self.__model(torch.cat((control, state), dim=1))

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
        running_baseline_loss = 0.0
        for controls, states, targets, dts in tqdm(train_loader, disable=not verbose, desc="Train", leave=False):
            controls, states, targets, dts = controls.float(), states.float(), targets.float(), dts.float()
            rollout_losses = []
            optimizer.zero_grad()
            cur_state = states[0]
            for state, control, target, dt in zip(states, controls, targets, dts):
                # acceleration * dt + prev state
                # m / s^2 * s + m / s
                new_state = model(control, cur_state) * dt.view(-1, 1)

                # 2 options:
                # 1. use true state
                # 2. use predicted state
                # second is probably better, since it takes whole rollout into account
                loss = criterion(new_state, target - cur_state)

                cur_state = new_state

                rollout_losses.append(loss)
            loss.sum().backward()
            optimizer.step()

            running_loss += loss.detach().cpu().item()
            # running_baseline_loss += criterion(y, x[:, -2:]).detach().cpu().item()

        train_loss = running_loss / len(train_loader)
        train_baseline_loss = running_baseline_loss / len(train_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/train baseline (zero acc.)", train_baseline_loss, epoch)
        desc = f"Epochs Train Loss {train_loss:.4g} (0 acc.) Loss {train_baseline_loss:.4g}"

        if val_loader is not None:
            with torch.no_grad():
                val_running_loss = 0.0
                val_running_baseline_loss = 0.0
                for x, y, dt in tqdm(val_loader, disable=not verbose, desc="Validation", leave=False):
                    y_pred = model(x) * dt

                    loss = criterion(y_pred, y - x[:, -2:])

                    val_running_loss += loss.detach().cpu().item()
                    val_running_baseline_loss += criterion(y, x[:, -2:]).detach().cpu().item()


            val_loss = val_running_loss / len(val_loader)
            val_baseline_loss = val_running_baseline_loss / len(val_loader)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("Loss/val baseline (zero acc.)", val_baseline_loss, epoch)
            desc += f" Val Loss {val_loss:.4g} (0 acc.) Loss {val_baseline_loss:.4g}"

        trange_epochs.set_description(desc)


def main():
    DELAY_STEPS = 3
    EPOCHS = 50
    TRAIN = True
    PLOT_VAL = True
    ROLLOUT_S = 10  # seconds
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
    model_name = f"models/example_hidden2relu_delay_{DELAY_STEPS}_model.pt"


    if TRAIN:
        model = Model()
        optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
        criterion = nn.MSELoss()

        train_sequences = load_bags("datasets/rzr_real", reader)
        train_dataset = SequenceLookaheadDataset(
            train_sequences, features, delayed_features, delay_steps=DELAY_STEPS, sequence_length=rollout_len
        )
        val_dataset = SequenceLookaheadDataset(val_sequences, features, delayed_features, delay_steps=DELAY_STEPS, sequence_length=rollout_len)

        print(len(train_dataset))
        print(len(val_dataset))

        train(
            model=model,
            optimizer=optimizer,
            train_loader=DataLoader(train_dataset, batch_size=32, shuffle=True),
            criterion=criterion,
            epochs=EPOCHS,
            # val_loader=DataLoader(val_dataset, batch_size=32, shuffle=True),
            verbose=True,
        )


if __name__ == "__main__":
    main()
