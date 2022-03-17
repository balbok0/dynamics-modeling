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
        for x, y, t in tqdm(train_loader, disable=not verbose, desc="Train", leave=False):
            optimizer.zero_grad()
            y_pred = model(x) * t

            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.detach().cpu().item()
        optimizer.zero_grad()

        train_loss = running_loss / len(train_loader)
        writer.add_scalar("Loss/train", train_loss, epoch)
        desc = f"Epochs Train Loss {train_loss:.4g}"

        if val_loader is not None:
            val_running_loss = 0.0
            for x, y in tqdm(val_loader, disable=not verbose, desc="Validation", leave=False):
                optimizer.zero_grad()
                y_pred = model(x)

                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()

                val_running_loss += loss.detach().cpu().item()

            val_loss = val_running_loss / len(val_loader)
            writer.add_scalar("Loss/val", val_loss, epoch)
            desc += f"Val Loss {val_loss:.4g}"

        trange_epochs.set_description(desc)