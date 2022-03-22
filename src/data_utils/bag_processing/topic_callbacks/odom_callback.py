import numpy as np
import rospy
from .abstract_callback import AbstractTopicCallback
from nav_msgs.msg import Odometry

class OdomCallback(AbstractTopicCallback):
    topics = ["/{robot_name}/odom"]
    feature = "state"

    def callback(self, msg: Odometry, ts: rospy.Time, current_state, *args, **kwargs):
        # Dictionaries are modified in place in python
        current_state[self.__class__.feature] = np.array([msg.twist.twist.linear.x, msg.twist.twist.angular.z])

        return False
