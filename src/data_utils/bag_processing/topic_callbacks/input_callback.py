from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np
import rospy
from .msg_stubs import VehicleInput, AutomaticGearDirectionDict
from .abstract_callback import AbstractTopicCallback

class InputCallback(AbstractTopicCallback):
    topics = ["/{robot_name}/input"]
    feature = "control"

    THRESHOLD_NEW_SEQUENCE = 0.15  # seconds. Time after which a new sequence is started.

    def __init__(self, features):
        super().__init__(features)
        self.required_features_set = set(self.required_features)  # This is to speed up check in the beggining of callback. It will run often
        self.current_sequence = defaultdict(list)
        self.last_input_ts = None

    def callback(
        self,
        msg: VehicleInput,
        ts: rospy.Time,
        current_state: Dict[str, np.ndarray],
        *args,
        **kwargs,
    ) -> Optional[Dict[str, List]]:
        result = None

        # Add vehicle input information to the current state.
        current_state[self.__class__.feature] = np.array(
            [
                msg.steer * AutomaticGearDirectionDict[msg.automatic_gear],
                msg.throttle * AutomaticGearDirectionDict[msg.automatic_gear],
                msg.brake,
            ]
        )


        # Do not modify anything we are waiting for state, costmap etc.
        if not self.required_features_set.issubset(current_state.keys()):
            return

        # TODO: Check if the input is the same as one before. If so continue.
        # Real vehicle spams /input message quite often (avg./median ~30 Hz, range 25-40Hz)
        # These seem to be quite different at every step. It might be that PID is modifying the steps.
        # It would be really beneficial to not have such an overlapping data.
        # However, this can make spacing of commands inconsistent across time, which can cause issues.

        # Check if this message is a start of a new sequence
        if self.last_input_ts is not None and (ts - self.last_input_ts).to_sec() > self.THRESHOLD_NEW_SEQUENCE:
            # If so, record it, clean current sequence and DO NOT start one immediately. This can cause boundary issues.
            result = self.current_sequence.copy()
            self.current_sequence = defaultdict(list)
            self.last_input_ts = ts
            return result

        self.last_input_ts = ts

        # All of the keys are found, and it still is the same sequence.
        # Record the sequence, in order
        for feature in self.required_features:
            self.current_sequence[feature].append(current_state[feature])
        # Also record time
        self.current_sequence["time"].append(ts.to_sec())

    def end_bag(self) -> Optional[Dict[str, List]]:
        return self.current_sequence
