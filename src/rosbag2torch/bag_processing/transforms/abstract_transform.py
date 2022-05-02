from abc import ABC, abstractmethod, abstractproperty, abstractstaticmethod
from typing import Dict, List, Set, Tuple

import numpy as np
import rospy


class AbstractTransform(ABC):
    @property
    @abstractstaticmethod
    def topics() -> List[Set[str]]:
        """
        An ordered list of sets of topics which given callback can process.
        """
        pass

    @abstractproperty
    def feature(self) -> str:
        """
        A string on what feature this callback will output.
        """
        pass

    def __call__(self, *args, **kwargs):
        return self.callback(*args, **kwargs)

    def __init__(self, features: List[str]):
        self.required_features = features

    @abstractmethod
    def callback(
        self,
        topic: str,
        msg,
        ts: rospy.Time,
        current_state: Dict[str, np.ndarray],
        *args,
        **kwargs,
    ):
        pass

    @abstractmethod
    def end_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def end_bag(self):
        return None
