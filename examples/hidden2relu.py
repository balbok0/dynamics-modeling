from rosbag2torch import load_bags, readers, filters, LookaheadSequenceDataset
from typing import Optional
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange


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
    reader = readers.ASyncSequenceReader(
        ["control", "state", "target"],
        features_to_record_on=["control"],
        filters=[
            filters.ForwardFilter(),
            filters.PIDInfoFilter()
        ]
    )

    model = nn.Sequential(
        nn.Linear(5, 32),
        nn.ReLU(),
        nn.Linear(32, 32),
        nn.ReLU(),
        nn.Linear(32, 2)
    )
    optimizer = optim.Adam(model.parameters(), weight_decay=0.01)
    criterion = nn.MSELoss()

    train_sequences = load_bags("datasets/rzr_real_bag", reader)
    print(train_sequences)
    train_dataset = LookaheadSequenceDataset(train_sequences, ["control", "state"], ["target"], delay_steps=1)

    val_sequences = load_bags("datasets/rzr_real_bag_val", reader)
    val_dataset = LookaheadSequenceDataset(val_sequences, ["control", "state"], ["target"], delay_steps=1)

    train(
        model,
        optimizer,
        DataLoader(train_dataset, batch_size=32, shuffle=True),
        criterion,
        epochs=500,
        val_loader=DataLoader(val_dataset, batch_size=32, shuffle=True),
        verbose=True,
    )

    torch.jit.script(model).save("example_hidden2relu_model.pt")


if __name__ == "__main__":
    main()
