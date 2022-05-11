from bisect import bisect_right
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .__defs import RawSequences


class SequenceLookaheadDataset(Dataset):
    def __init__(
        self,
        sequences: RawSequences,
        features_with_delays: List[Tuple[str, int]],
        sequence_length: int = 50,
    ) -> None:

        # Sanitize features (i.e. make sure that minimum delay is 0)
        min_delay = min(delay for _, delay in features_with_delays)
        features_with_delays = [(f, delay - min_delay) for f, delay in features_with_delays]

        # Set all of the variables
        self.__features = features_with_delays
        self.__sequence_length = sequence_length

        (
            self.processed_sequences,
            self.__sequence_lengths,
            self.__max_len_rollout_idx,
            self.__max_len_rollout,
        ) = self.__class__.__parse_sequences(
            sequences,
            features=self.__features,
            sequence_length=sequence_length,
        )
        # Index of first rollout in each sequence
        self.__sequence_start_idxs = np.cumsum([0] + self.__sequence_lengths[:-1])
        # Total number of rollouts in all sequences
        self.__total_len = sum(self.__sequence_lengths)

    def __len__(self) -> int:
        return self.__total_len

    def __get_sequence_of_length(
        self, index: int, sequence_length: int
    ) -> Tuple[torch.Tensor, ...]:
        sequence_idx = bisect_right(self.__sequence_start_idxs, index) - 1  # type: ignore
        rollout_idx = index - self.__sequence_start_idxs[sequence_idx]

        result: List[torch.Tensor] = []
        for f, delay in self.__features:
            result.append(self.processed_sequences[sequence_idx][f"{f}_{delay}"][rollout_idx:rollout_idx+sequence_length])
            result.append(self.processed_sequences[sequence_idx][f"time_{delay}"][rollout_idx:rollout_idx+sequence_length])

        return tuple(result)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        return self.__get_sequence_of_length(index, self.__sequence_length)

    @property
    def longest_rollout(self) -> Tuple[torch.Tensor, ...]:
        return self.__get_sequence_of_length(
            self.__sequence_start_idxs[self.__max_len_rollout_idx],
            self.__max_len_rollout,
        )

    @staticmethod
    def __parse_sequences(
        sequences: RawSequences,
        features: List[Tuple[str, int]],
        sequence_length: int,
        *args,
        **kwargs,
    ):
        processed_sequences: List[Dict[str, torch.Tensor]] = []
        sequence_lengths: List[int] = []
        max_len_rollout = 0
        max_len_rollout_idx = 0

        max_delay = max(delay for _, delay in features)

        required_features = set(f for f, _ in features)

        seq_idx = 0

        for seq in sequences:
            if not required_features.issubset(seq.keys()):
                # Sequence does not contain all required features
                continue

            cur_seq = {}

            # Add torch sequences
            for f, delay in features:
                if delay == max_delay:
                    cur_seq[f"{f}_{delay}"] = torch.from_numpy(seq[f][delay:])
                    cur_seq[f"time_{delay}"] = torch.from_numpy(seq["time"][delay:] - seq["time"][:-delay])
                else:
                    cur_seq[f"{f}_{delay}"] = torch.from_numpy(seq[f][delay:-(max_delay - delay)])
                    cur_seq[f"time_{delay}"] = torch.from_numpy(seq["time"][delay:-(max_delay - delay)] - seq["time"][:-max_delay])

            # Check key is used for checking length of this sequence etc.
            # It should not be assumed to be a specific feature
            check_key = next(iter(cur_seq.keys()))

            # Calculate the number of sequences that can be read
            # In each step we are taking delay_steps steps forward (to get next element of the rollout)
            # for sequence_length steps
            # One rollout will be sequence_length long
            num_rollouts = (
                len(cur_seq[check_key]) - sequence_length + 1
            )

            # Sequence too short. No rollouts can be read
            if num_rollouts <= 0:
                continue

            # Add sequence to processed sequences
            processed_sequences.append(cur_seq)

            # Add it's length to the sequence lengths
            sequence_lengths.append(num_rollouts)

            # Determine whether it is the longest sequence
            if max_len_rollout < len(cur_seq[check_key]):
                max_len_rollout = len(cur_seq[check_key])
                max_len_rollout_idx = seq_idx

            # Only update seq_idx if we have a valid sequence
            seq_idx += 1

        return (
            processed_sequences,
            sequence_lengths,
            max_len_rollout_idx,
            max_len_rollout,
        )
