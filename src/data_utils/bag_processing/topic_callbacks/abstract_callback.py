import rospy

from abc import abstractmethod, ABC, abstractproperty
from typing import Dict, List, Optional


class AbstractTopicCallback(ABC):

    @abstractproperty
    def topics(self) -> List[str]:
        """
        An ordered list of topics which given callback can process.
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
        msg,
        ts: rospy.Time,
        current_state: Dict,
        *args,
        **kwargs,
    ) -> bool:
        pass

    def end_bag(self) -> bool:
        return None
