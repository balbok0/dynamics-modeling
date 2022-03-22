from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, List, Optional, Union

import numpy as np
from ..topic_callbacks import get_topics_and_callbacks

import rospy
import rosbag
from tqdm import tqdm
from scipy import interpolate


class FixedTimestepReader():
    def __init__(
        self,
        required_keys: List[str],
        max_in_seq_delay: Union[rospy.Duration, float] = 0.1,  # At least 10 Hz for messages,
        log_interval: Union[rospy.Duration, float] = 1 / 30,  # Around 30 Hz
        *args,
        **kwargs,
    ) -> None:
        self.sequences = []

        # Currently processed sequence
        self.cur_sequence_ts: DefaultDict[str, List[rospy.Time]] = defaultdict(list)
        self.cur_sequence_states: DefaultDict[str, List[Any]] = defaultdict(list)

        # Last processed timestamp
        self.last_ts: Optional[rospy.Time] = None

        # Required keys (features/after processing)
        self.required_keys = required_keys
        self.required_keys_set = set(required_keys)

        # Threshold at which to break the sequence
        self.THRESHOLD_NEW_SEQUENCE = max_in_seq_delay if isinstance(max_in_seq_delay, rospy.Duration) else rospy.Duration(secs=max_in_seq_delay)

    @staticmethod
    def interpolate_data(y, t, spline_pwr: int = 3):
        knots = np.linspace(t[0], t[-1], t.size/10.0)[1:-1]
        spline_params = interpolate.splrep(t, y, k = spline_pwr, t=knots)
        return spline_params


    def __finish_sequence(self):
        if len(self.cur_sequence_ts) <= 1:
            # TODO: Better error
            raise ValueError("Sequence not long enough")

        # Get the first processed timestamp
        min_ts = min([x[0] for x in self.cur_sequence_ts.values()])
        # Get the last timestamp
        max_ts = max([x[-1] for x in self.cur_sequence_ts.values()])

        cur_sequence_interpolated = {
            k: interpolate(
                np.array(self.cur_sequence_states[k]),
                np.array(self.cur_sequence_ts[k]) - min_ts
            )
            for k in self.cur_sequence_ts.keys()
        }

        # Lookup table of last processed ts index for each feature
        last_idx_ts = {k: 0 for k in self.cur_sequence_ts.keys()}

        ts = min_ts
        cur_sequence = defaultdict(list)
        while ts <= max_ts:
            for feature in self.cur_sequence_ts.keys():
                idx = last_idx_ts[feature]
                while (
                    idx < len(self.cur_sequence_ts[feature]) and
                    idx
                ):
                    last_idx_ts[feature] = idx
            pass
        pass

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
            if self.last_ts is not None and ts - self.last_ts > self.THRESHOLD_NEW_SEQUENCE:
                self.__finish_sequence()

            for callback in topics_with_callbacks[topic]:
                callback.callback(msg, ts, current_state)

                # Log timestamp and current state
                self.cur_sequence_ts[callback.feature].append(ts)
                self.cur_sequence_states[callback.feature].append(current_state[callback.feature])

            self.last_ts = ts


        for callback_arr in topics_with_callbacks.values():
            for callback in callback_arr:
                callback.end_bag()

        self.end_bag()

        return result
