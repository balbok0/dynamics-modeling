from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from tqdm import tqdm
from ..topic_callbacks import get_topics_and_callbacks

import rospy
import rosbag


class ASyncSequenceReader():
    def __init__(
        self,
        required_keys: List[str],
        max_in_seq_delay: Union[rospy.Duration, float] = 0.1,  # 10 Hz
        *args,
        **kwargs,
    ) -> None:
        self.sequences = []
        self.cur_sequence = defaultdict(list)
        self.last_ts: Optional[rospy.Time] = None
        self.required_keys = required_keys
        self.required_keys_set = set(required_keys)
        self.THRESHOLD_NEW_SEQUENCE = max_in_seq_delay if isinstance(max_in_seq_delay, rospy.Duration) else rospy.Duration(secs=max_in_seq_delay)

    def verify_sequence(self):
        # Verify that:
        #   1. All of requested features (and "time") are present in a sequence
        #   2. Lengths of sequence agree for all key-words
        if not (self.required_keys_set | {"time"}).issubset(self.cur_sequence.keys()):
            return False

        len_seq = len(self.cur_sequence["time"])
        return all([len(self.cur_sequence[k]) == len_seq for k in self.required_keys_set])

    def __finish_sequence(self):
        if len(self.cur_sequence) > 0:
            if not self.verify_sequence():
                raise ValueError(
                    f"Received invalid sequence. Features: {self.required_keys}, but keys of sequence are: {self.cur_sequence.keys()}"
                    "\nIt these match it probably is an issue with shapes of these features."
                )
            for k in self.cur_sequence.keys():
                self.cur_sequence[k] = np.asarray(self.cur_sequence[k])
            self.sequences.append(self.cur_sequence)
        # Initialize cur_sequence to defaultdict
        self.cur_sequence = defaultdict(list)

    def record_state(self, state: Dict[str, Any], ts: rospy.Time):
        if not self.required_keys_set.issubset(state.keys()):
            # Not all keys are inserted yet
            return

        # TODO: Check if the input is the same as one before. If so continue.
        # Real vehicle spams /input message quite often (avg./median ~30 Hz, range 25-40Hz)
        # These seem to be quite different at every step. It might be that PID is modifying the steps.
        # It would be really beneficial to not have such an overlapping data.
        # However, this can make spacing of commands inconsistent across time, which can cause issues.

        # Check if this message is a start of a new sequence
        if self.last_ts is not None and ts - self.last_ts > self.THRESHOLD_NEW_SEQUENCE:
            self.__finish_sequence()
            self.last_input_ts = ts
            return

        self.last_input_ts = ts

        # All of the keys are found, and it still is the same sequence.
        # Record the sequence, in order
        for feature in self.required_keys:
            self.cur_sequence[feature].append(state[feature])

        # Also record time
        self.cur_sequence["time"].append(ts.to_sec())

    def end_bag(self):
        self.__finish_sequence()
        self.last_ts = None

    def bag_extract_data(self, bag_file_path: Union[str, Path]):
        bag_file_path = Path(bag_file_path)
        bag = rosbag.Bag(bag_file_path)
        topics = bag.get_type_and_topic_info().topics

        # Get robot name. It should be the most often occuring topmost key
        topic_parents, topic_parents_counts =  np.unique([x.split("/", 2)[1] for x in topics.keys()], return_counts=True)
        robot_name = topic_parents[np.argmax(topic_parents_counts)]

        topics_with_callbacks = get_topics_and_callbacks(
            self.required_keys, set(topics.keys()), {"robot_name": robot_name}
        )

        # Result holds a list of dictionaries. One for each sequence.
        result = []

        # Current state is modified in-place
        current_state = {}
        for topic, msg, ts in tqdm(
            bag.read_messages(topics=topics_with_callbacks.keys()),
            total=sum([topics[k].message_count for k in topics_with_callbacks.keys()]),
            desc=f"Extracting from bag: {bag_file_path.name}"
        ):
            log_at_ts = False
            for callback in topics_with_callbacks[topic]:
                log_at_ts |= callback.callback(msg, ts, current_state)

            if log_at_ts:
                self.record_state(current_state, ts)

        for callback_arr in topics_with_callbacks.values():
            for callback in callback_arr:
                callback.end_bag()

        self.end_bag()

        return result
