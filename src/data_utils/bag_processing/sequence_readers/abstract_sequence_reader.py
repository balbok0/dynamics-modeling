from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
import hashlib
import os
from pathlib import Path

import inspect
import h5py
import numpy as np
from tqdm import tqdm
from ..filters import AbstractFilter, get_filters_topics
import rospy
import rosbag
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from ..transforms import get_topics_and_transforms


# Each out feature points to a list of values. They have to have the same length.
# There is also required "time" key (see verify_sequence method) which is timestamp (float, in seconds) of recorded state.
Sequence = Dict[str, List]  # With required "time" key


# Similar to Sequence, but there is no time key, and lengths can differ.
# Time is per-feature rather than per-state.
# This enables for things like looking up the last state message before a certain time, or interpolations etc.
RawSequence = Dict[str, Tuple[List, List]]  # Similar


# Multiple Sequences. Return type to main process.
Sequences = List[Sequence]


# Hashing utility
# Source: 
# BUF_SIZE is totally arbitrary, change for your app!
BUF_SIZE = 65536  # lets read stuff in 64kb chunks!

def hash_file(path: Path) -> str:
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        while True:
            data = f.read(BUF_SIZE)
            if not data:
                break
            md5.update(data)
    return md5.hexdigest()


class AbstractSequenceReader(ABC):
    def __init__(
        self,
        required_keys: List[str],
        filters: List[AbstractFilter] = [],
        *args,
        **kwargs,
    ) -> None:
        """AbstractSequenceReader is a class that combines filters and topic transforms in order to read a bag file.
        While it itself does not provide a specific output (see it's subclasses), it provides methods that can be used in such process.

        Args:
            required_keys (List[str]): Required output keys of a sequence.
        """
        self.required_keys = required_keys
        self.required_keys_set = set(required_keys)

        self.filters = filters

        self.end_bag()

    @abstractproperty
    def sequences(self) -> Sequences:
        pass

    def __write_raw_sequences_to_cache(self, bag_file_path: Path, transforms: Set[AbstractFilter]):
        # First get the hash. It only depends on filters used (self.filters).
        # NOTE: These are raw sequences only, so we don't need to include transforms, meaning we only depend on logic in the abstract_sequence_reader.
        class_filter_hash = hashlib.md5(" ".join([f.__class__.__name__ for f in self.filters]).encode("utf-8")).hexdigest()
        cache_file_path = bag_file_path.parent / f"{bag_file_path.stem}_{class_filter_hash}_.h5cache"

        with h5py.File(cache_file_path, "w") as hf:
            # First write the meta keys.
            hf_meta = hf.create_group("meta")

            # First hash the abstract sequence reader itself.
            hf_meta["sequence_reader_hash"] = hash_file(__file__)

            # Then hash the filters and transf.
            for f_ in set(self.filters) | transforms:
                key = f_.__class__.__name__
                hf_key = hf_meta.create_group(key)

                key_path = inspect.getsourcefile(f_.__class__)
                common_prefix = os.path.commonprefix([key_path, __file__])
                file_path = os.path.relpath(key_path, Path(__file__).parent)
                hf_key["file_path"] = file_path
                hf_key["file_hash"] = hash_file(Path(__file__).parent / file_path)

            # Then write the raw sequences.
            for sequence_idx, sequence in enumerate(self.cur_bag_raw_sequences):
                hf_sequence = hf.create_group(f"sequence_{sequence_idx}")
                for k, v in sequence.items():
                    hf_sequence.create_group(k)
                    hf_sequence[k]["data"] = v[0]
                    hf_sequence[k]["time"] = v[1]

    def __read_raw_sequences_from_cache(self, bag_file_path: Path) -> Optional[List[RawSequence]]:
        """
        Check if bag file has already been processed. If so, return cached sequences.
        If not, return None.
        """

        # First get the hash. It only depends on filters used (self.filters).
        # NOTE: These are raw sequences only, so we don't need to include transforms, meaning we only depend on logic in the abstract_sequence_reader.
        class_filter_hash = hashlib.md5(" ".join([f.__class__.__name__ for f in self.filters]).encode("utf-8")).hexdigest()
        cache_file_path = bag_file_path.parent / f"{bag_file_path.stem}_{class_filter_hash}_.h5cache"

        # If cache doesn't exist, return None.
        if not cache_file_path.exists():
            return None

        with h5py.File(cache_file_path, "r") as hf:
            # If meta key doesn't exist, then it's an old cache file.
            if "meta" not in hf:
                return None

            # Verify that the cache file has subset of the required keys.
            if not (self.required_keys_set).issubset(hf["meta"].keys()):
                return None

            # Verify that the cache file has subset of the filters as well
            if not set([f.__class__.__name__ for f in self.filters]).issubset(hf["meta"].keys()):
                return None

            for key in self.required_keys_set | set([f.__class__.__name__ for f in self.filters]):
                # For each feature key, check what is the hash of the file that generates it.
                key_path = Path(__file__).parent / hf["meta"][key]["file_path"]
                key_hash = hf["meta"][key]["file_hash"]

                # If the feature doesn't exist here than it's a new feature, or something else changed.
                if not key_path.exists():
                    return None

                cur_hash = hash_file(key_path)

                # If the hash doesn't match, then the file has changed.
                if key_hash != cur_hash:
                    return None

            # Lastly check whether this file is newer than the cached version.
            if hf["meta"]["sequence_reader_hash"] != hash_file(__file__):
                return None

            # All of the checks passed, so we can load the cache
            sequences = []
            for sequence_key in filter(lambda x: x.startswith("sequence"), hf.keys()):
                # If required keys are not in the sequence, then it's not a valid sequence.
                # Probably something got corrupted during saving of cache. Return None
                if not (self.required_keys_set).issubset(hf[sequence_key].keys()):
                    return None

                hf_sequence = hf[sequence_key]
                sequence = {}
                for feature_key in self.required_keys_set:
                    sequence[feature_key] = (
                        np.array(hf_sequence[feature_key]["data"]),
                        np.array(hf_sequence[feature_key]["time"]),
                    )
                sequences.append(sequence)

        return sequences

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

    def _extract_raw_sequences(self, bag_file_path: Union[str, Path]):
        bag_file_path = Path(bag_file_path)

        # First check cache
        cached_result = self.__read_raw_sequences_from_cache(bag_file_path)
        if cached_result is not None:
            self.cur_bag_raw_sequences = cached_result
            return

        bag = rosbag.Bag(bag_file_path)
        topics = bag.get_type_and_topic_info().topics

        # Get robot name. It should be the most often occuring topmost key
        topic_parents, topic_parents_counts =  np.unique([x.split("/", 2)[1] for x in topics.keys()], return_counts=True)
        robot_name = topic_parents[np.argmax(topic_parents_counts)]

        topics_with_transforms = get_topics_and_transforms(
            self.required_keys, set(topics.keys()), {"robot_name": robot_name}
        )

        if self.filters == []:
            filtered_ts = [(rospy.Time(bag.get_start_time()), rospy.Time(bag.get_end_time()))]
        else:
            filtered_ts = []
            filter_topics = get_filters_topics(self.filters, set(topics.keys()), {"robot_name": robot_name})

            last_log_state = False
            cur_start_time = None


            for topic, msg, ts in tqdm(
                bag.read_messages(topics=filter_topics.keys()),
                desc="Filtering bag",
                leave=False,
            ):
                # Callback for current topic
                for filter_ in filter_topics[topic]:
                    filter_.callback(msg, ts, topic)

                # Whether all filters agree that current time should be logged
                cur_log_state = all([f.should_log for f in self.filters])

                # If state changed either log start time of this chunk or end current chunk
                if cur_log_state and not last_log_state:
                    cur_start_time = ts
                elif not cur_log_state and last_log_state:
                    filtered_ts.append((cur_start_time, ts))

                # Update last_log_state
                last_log_state = cur_log_state

        # Get all of the transforms
        transforms: Set[AbstractFilter] = set()
        seen_transform_features: Set[str] = set()
        for topic_transforms in topics_with_transforms.values():
            for transform in topic_transforms:
                if transform.feature not in seen_transform_features:
                    transforms.add(transform)
                    seen_transform_features.add(transform.feature)

        # Call transforms/handlers to process data for each sequence.
        for ts_start, ts_end in filtered_ts:
            current_state = {}
            for topic, msg, ts in tqdm(
                bag.read_messages(topics=topics_with_transforms.keys(), start_time=ts_start, end_time=ts_end),
                desc=f"Extracting transforms from bag: {bag_file_path.name}",
                leave=False,
            ):
                # Callback for current topic
                for transform in topics_with_transforms[topic]:
                    transform.callback(msg, ts, current_state)

            for transform in transforms:
                self.cur_raw_sequence[transform.feature] = transform.end_sequence()

            # Add current sequence to list of sequences
            self.cur_bag_raw_sequences.append(self.cur_raw_sequence)

            # Reset current sequence
            self.cur_raw_sequence = defaultdict(lambda: ([], []))

        # Write cache
        self.__write_raw_sequences_to_cache(bag_file_path, transforms)

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
        """Ends bag. Cleans up the current state etc.
        It should be called at the end of each bag file by the subclass.
        """
        self.cur_bag_raw_sequences: List[RawSequence] = []
        self.cur_raw_sequence = defaultdict(lambda: ([], []))
