from collections import defaultdict
from email.policy import default
from typing import List

import rospy
from .abstract_callback import AbstractTopicCallback
from nav_msgs.msg import Odometry
import numpy as np
import scipy.spatial.transform as trf

class AutorallyGroundTruth(AbstractTopicCallback):
    topics = ["/{robot_name}/odom"]
    feature = "autorally-ground-truth"

    def __init__(self, features: List[str], use_quarterions: bool = True):
        """Ground Truth based on Autorally project.
        """
        super().__init__(features)

        self.previous_pose = None
        self.use_quarterions = use_quarterions

        self.history_poses = defaultdict(list)
        self.history_ts = defaultdict(list)

    def callback(self, msg: Odometry, ts: rospy.Time, current_state, *args, **kwargs):
        # Get angle of pose
        angle = np.array([
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ])
        if self.use_quarterions:
            angle = trf.Rotation(angle).as_euler("xyz")
            ya = angle[-1]
        else:
            ya = trf.Rotation(angle).as_euler("xyz")[-1]

        # Get velocity
        rot_mat = np.array([[np.cos(ya), np.sin(ya)], [-np.sin(ya), np.cos(ya)]])
        vel_wf = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        vel_bf = np.dot(rot_mat, vel_wf)
        vel_x = vel_bf[0]
        vel_y = vel_bf[1]

        heading_rate = msg.twist.twist.angular.z

        current_state[self.__class__.feature] = np.array(
            [*angle, vel_x, vel_y, heading_rate]
        )

        return True, ts

    def end_bag(self) -> bool:

        # Fit spline through the 

        # Reset history
        self.history_poses = defaultdict(list)
        self.history_ts = defaultdict(list)
