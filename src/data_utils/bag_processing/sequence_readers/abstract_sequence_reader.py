from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from pathlib import Path
import rospy
import rosbag
from typing import Any, Dict, List, Optional, Tuple, Union


# Each out feature points to a list of values. They have to have the same length.
# There is also required "time" key (see verify_sequence method) which is timestamp (float, in seconds) of recorded state.
Sequence = Dict[str, List]  # With required "time" key


# Similar to Sequence, but there is no time key, and lengths can differ.
# Time is per-feature rather than per-state.
# This enables for things like looking up the last state message before a certain time, or interpolations etc.
RawSequence = Dict[str, Tuple[List, List]]  # Similar


# Multiple Sequences. Return type to main process.
Sequences = List[Sequence]


class AbstractSequenceReader(ABC):
    def __init__(
        self,
        required_keys: List[str],
        *args,
        **kwargs,
    ) -> None:
        """AbstractSequenceReader is a class that combines filters and topic callbacks in order to read a bag file.
        While it itself does not provide a specific output (see it's subclasses), it provides methods that can be used in such process.

        Args:
            required_keys (List[str]): Required output keys of a sequence.
        """
        self.cur_raw_sequence = defaultdict(([], []))
        self.cur_bag_raw_sequences: List[RawSequence] = []

        self.required_keys = required_keys
        self.required_keys_set = set(required_keys)

    @abstractproperty
    def sequences(self) -> Sequences:
        pass

    @abstractmethod
    def extract_bag_data(self, bag_file_path: Union[str, Path]):
        """Extract Bag Data.
        This method will be called by the main process.
        It's purpose is to populate the self.sequences list with the sequences from the provided bag file.

        Note:
            - A lot of subclasses of AbstractSequenceReader might want to use self._extract_raw_sequences,
                which populates self.cur_bag_raw_sequences with the raw sequences from the bag file.

        Args:
            bag_file_path (Union[str, Path]): Path to the bag file.
        """
        pass

    def verify_sequence(self, sequence: Sequence) -> bool:
        """
        Verify that:
            1. All of requested features (and "time") are present in a sequence.
            2. Lengths of sequence agree for all key-words.

        Returns:
            bool: True if above conditions are met, False otherwise.
        """
        if not (self.required_keys_set | {"time"}).issubset(self.cur_sequence.keys()):
            return False

        len_seq = len(self.cur_sequence["time"])
        return all([len(self.cur_sequence[k]) == len_seq for k in self.required_keys_set])

    def end_bag(self):
        self.cur_bag_raw_sequences = []
        self.cur_raw_sequence = defaultdict(([], []))
