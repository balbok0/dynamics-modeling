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
    ) -> bool:
        result = None

        # Add vehicle input information to the current state.
        current_state[self.__class__.feature] = np.array(
            [
                msg.steer * AutomaticGearDirectionDict[msg.automatic_gear],
                msg.throttle * AutomaticGearDirectionDict[msg.automatic_gear],
                msg.brake,
            ]
        )

        return False

