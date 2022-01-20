from .abstract_model import AbstractNumpyModel, WEIGHTS
import numpy as np

class LinearModel(AbstractNumpyModel):
    name = "linear"
    features = ["state", "control", "target"]

    def train(self, seqs, n_steps=1):
        self.train_n_steps = n_steps
        if n_steps != 1:
            print("Warning: Linear model training with n_steps > 1 is not supported!")
        train_x, train_y = self.concat_seqs(seqs, n_steps)
        weighted_y = train_y * WEIGHTS
        assert(weighted_y.shape == train_y.shape)
        assert(len(train_x) == len(train_y))
        weighted_w = np.linalg.solve(
            train_x.T @ train_x + self.lambda_ * np.eye(train_x.shape[1]),
            train_x.T @ weighted_y,
        )
        print(f"Train x: {train_x.shape}")
        print(f"Train y: {train_y.shape}")


class MeanModel(AbstractNumpyModel):
    name = "mean"
    features = ["state", "control", "target"]

    def __init__(self, D, H, P):
        super().__init__(D, H, P, 0)
        self.mean = None
        self.train_n_steps = 1

    def train(self, seqs, n_steps=1):
        train_x, train_y = self.concat_seqs(seqs, n_steps)
        self.mean = train_y.mean(axis=0)
        assert(self.mean.shape == (train_y.shape[1],))

    def predict_one_steps(self, xx):
        return np.tile(self.mean, (xx.shape[0], 1))

class UnicycleModel(AbstractNumpyModel):
    name = "unicycle"
    features = ["state", "control", "target"]

    def __init__(self, D, H, P, delay_steps):
        # Unicycle model doesn't account for velocity state (features 2 and 3)
        super().__init__(D, H, P, delay_steps)
        self.train_n_steps = 1

    def train(self, seqs, n_steps=1):
        dt = 0.1
        effective_wheel_base = 1.80
        measured_wheel_base = 1.08
        theta_factor = measured_wheel_base / effective_wheel_base
        self.w = dt * np.array([
            [1, 0, 0],
            [0, 0, theta_factor]
            ])

class GTTwistModel(AbstractNumpyModel):
    name = "gt_twist"
    features = ["state", "control", "target"]

    def __init__(self, D, H, P):
        super().__init__(D, H, P, 0)
        self.features_to_use = np.zeros(D+H+P, dtype=np.bool)
        self.features_to_use[-3:] = True
        self.train_n_steps = 1

        # Before training, set a default
        dt = 0.1
        self.w = dt * np.eye(3)
