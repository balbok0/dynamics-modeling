import torch
from torch import nn

from ..abstract_model import ModelWithSpec


class Linear(nn.Module, ModelWithSpec):
    name = "linear"
    x_features = ["state", "control"]
    y_features = ["target"]

    dataset_name = "torch_lookahead"
    opt_algo = "regression_train_loop"

    def __init__(self) -> None:
        super().__init__()

        # dim_in: 2 (state) + 3 (control)
        # dim_out: 3 (state displacement)
        self.layer = nn.Linear(5, 3)

    def forward(self, x):
        return self.layer(x)