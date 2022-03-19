from typing import List
import numpy as np
import rospy
from .abstract_callback import AbstractTopicCallback
from nav_msgs.msg import Odometry
from ...general_utils import planar_pose, planar_twist
import scipy.spatial.transform as trf


class GroundTruthCallback(AbstractTopicCallback):
    topics = ["/unity_command/ground_truth/{robot_name}", "/{robot_name}/odom"]
    feature = "target"

    def __init__(self, features: List[str]):
        super().__init__(features)

        self.previous_pose = None

    def callback(self, msg: Odometry, ts: rospy.Time, current_state, *args, **kwargs):
        z_angle = trf.Rotation([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]).as_euler("zyx")[0]

        current_state[self.__class__.feature] = np.array([msg.pose.pose.position.x, msg.pose.pose.position.y, z_angle])
