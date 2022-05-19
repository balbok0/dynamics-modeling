from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from rosbag2torch.datasets.torch_sequence_lookahead import SequenceLookaheadDataset

from .__defs import RawSequences


class LookaheadDataset(Dataset):
    def __init__(
        self,
        sequences: RawSequences,
        features_with_delays: List[Tuple[str, int]],
    ) -> None:
        super().__init__()

        self.__sequence_dataset = SequenceLookaheadDataset(
            sequences=sequences,
            features_with_delays=features_with_delays,
            sequence_length=1,
        )

    def __len__(self) -> int:
        return len(self.__sequence_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        result: List[torch.Tensor] = []
        for feature_tensor in self.__sequence_dataset[index]:
            # "squeeze" the second (sequence) dimension
            # Since sequence length is 1 it's going to be of length 1
            result.append(feature_tensor[0])
        return tuple(result)

    @property
    def longest_rollout(self) -> Tuple[torch.Tensor, ...]:
        return self.__sequence_dataset.longest_rollout
