from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from .named_dataset import NamedDataset
from torch.utils.data import Dataset

class LookaheadSequenceDataset(Dataset, NamedDataset):
    name = "torch_lookahead"

    def __init__(
        self,
        seqs: List[Dict[str, np.ndarray]],
        x_features: List[str],
        y_features: List[str],
        delay_steps: int = 1,
        n_steps: int = 1,
    ) -> None:
        super().__init__()

        x, y, t = self.__class__._rollout_sequences(seqs, x_features, y_features, delay_steps, n_steps)

        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.t = torch.from_numpy(t)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index], self.t[index]

    @staticmethod
    def _relative_pose(query_pose: np.ndarray, reference_pose: np.ndarray) -> np.ndarray:
        diff = query_pose - reference_pose
        distance = np.linalg.norm(diff[:, :2], axis=1)
        direction = np.arctan2(diff[:, 1], diff[:,0])
        relative_direction = direction - reference_pose[:, 2]
        angle_diff = diff[:, 2]
        minimized_angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))
        return np.array([
            distance*np.cos(relative_direction),
            distance*np.sin(relative_direction),
            minimized_angle_diff,
        ]).T

    @staticmethod
    def _rollout_sequences(
        seqs: List[Dict[str, np.ndarray]],
        x_features: List[str],
        y_features: List[str],
        delay_steps: int,
        n_steps: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # Pre-allocate arrays, get indexes of corresponding features
        seqs_len = 0
        x_feature_size = 0
        y_feature_size = 0
        for s in seqs:
            seqs_len += len(s[x_features[0]]) - n_steps - delay_steps

        x_seqs = None
        y_seqs = None
        t_seqs = np.zeros((seqs_len,), dtype=np.float32)

        # Process data
        seqs_so_far = 0
        for s in seqs:
            # Get data for sequence
            s_len = len(s[x_features[0]]) - n_steps - delay_steps
            for a in [
                s[f][:s_len] for f in x_features
            ]:
                print(a.shape)
            x_s = np.concatenate([
                s[f][:s_len] for f in x_features
            ], axis=1)
            y_s = np.concatenate([
                s[f] for f in y_features
            ], axis=1)

            if "time" in s:
                t_s = s["time"][n_steps + delay_steps:] - s["time"][:s_len]
            else:
                t_s = np.ones(s_len)

            # Get target y
            relative_targets = __class__._relative_pose(y_s[n_steps:], y_s[:n_steps])
            y_s = relative_targets[delay_steps:]

            # If it's a first sequence define x_seqs and y_seqs with proper shapes
            if x_seqs is None:
                x_seqs = np.zeros((seqs_len, x_s.shape[1]), dtype=np.float32)
                y_seqs = np.zeros((seqs_len, y_s.shape[1]), dtype=np.float32)

            # Append to result
            x_seqs[seqs_so_far:seqs_so_far + s_len] = x_s
            y_seqs[seqs_so_far:seqs_so_far + s_len] = y_s
            t_seqs[seqs_so_far:seqs_so_far + s_len] = t_s
            seqs_so_far += s_len

        return x_seqs, y_seqs, t_seqs
