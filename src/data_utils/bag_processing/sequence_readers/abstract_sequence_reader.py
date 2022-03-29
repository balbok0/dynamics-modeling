from abc import ABC, abstractmethod, abstractproperty
from collections import defaultdict
from pathlib import Path

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
                total=sum([topics[k].message_count for k in filter_topics.keys()]),
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
                    if (ts - cur_start_time).to_sec() < 5.0:
                        for filter_ in self.filters:
                            print(f"Filter {filter_.name}: {filter_.should_log}")
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
            print(f"duration: {(ts_end - ts_start).to_sec()}")
            current_state = {}
            for topic, msg, ts in tqdm(
                bag.read_messages(topics=topics_with_transforms.keys(), start_time=ts_start, end_time=ts_end),
                desc=f"Extracting transforms from bag: {bag_file_path.name}",
                total=sum([topics[k].message_count for k in topics_with_transforms.keys()]),
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
