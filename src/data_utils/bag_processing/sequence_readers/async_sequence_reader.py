from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from tqdm import tqdm

from .abstract_sequence_reader import AbstractSequenceReader, Sequence
from ..transforms import get_topics_and_transforms
from ..filters import AbstractFilter

import rospy
import rosbag


class ASyncSequenceReader(AbstractSequenceReader):
    def __init__(self, required_keys: List[str], features_to_record_on: List[str], filters: List[AbstractFilter] = [],  *args, **kwargs) -> None:
        super().__init__(required_keys, filters, *args, **kwargs)

        assert len(features_to_record_on) > 0, "features_to_record_on must be a non-empty list"
        assert set(features_to_record_on).issubset(set(required_keys))

        self._sequences = []

        self.features_to_record_on = []

    @property
    def sequences(self):
        return self._sequences

    def _transform_raw_sequences(self):
        print(f"Raw sequences gotten: {len(self.cur_bag_raw_sequences)}")
        for raw_sequence in self.cur_bag_raw_sequences:
            print("Here 1")

            # Set is used to ensure timestamps are unique
            tmp = set()
            # Raw sequence keys
            print(f"Raw sequence: {list(raw_sequence.keys())}")
            print(f"Features to record on: {self.features_to_record_on}")
            for feature_name in self.features_to_record_on:
                tmp.update(raw_sequence[feature_name][1])
            print(f"tmp len: {len(tmp)}")
            if not tmp:
                # If the sequence is empty, ignore it
                continue
            ts_to_record_on = np.sort(tmp)

            print("Here 2")

            # Traverse sequentially through timestamps to be logged on
            # Ensure that all of the required keys are present at the time of logging
            # and if so log the state at the time
            last_used_idx = {
                feature_name: -1
                for feature_name in self.required_keys
            }
            sequence: Sequence = {
                feature_name: []
                for feature_name in self.required_keys
            }
            for ts in ts_to_record_on:
                # For each feature get the latest index that is less than or equal to the current timestamp
                for feature in self.required_keys:
                    while (
                        last_used_idx[feature] + 1 != len(raw_sequence[feature][1])
                        and raw_sequence[feature][1][last_used_idx[feature] + 1] <= ts
                    ):
                        last_used_idx[feature] += 1

                # If any of the required features still didn't occur, break
                if any(last_used_idx[feature] == -1 for feature in self.required_keys):
                    continue

                # Add current state to the sequence
                for feature in self.required_keys:
                    sequence[feature].append(raw_sequence[feature][0][last_used_idx[feature]])

                # Lastly add time
                sequence["time"].append(ts)

            # Add sequence to the list of sequences, if it's non-empty
            if len(sequence) > 0:
                self.sequences.append(sequence)

    def extract_bag_data(self, bag_file_path: Union[str, Path]):
        self._extract_raw_sequences(bag_file_path)
        self._transform_raw_sequences()
        self.end_bag()