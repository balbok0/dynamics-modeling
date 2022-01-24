import numpy as np
import rospy
from .abstract_callback import AbstractTopicCallback
from nav_msgs.msg import Odometry
from ...general_utils import planar_pose, planar_twist


class GroundTruthCallback(AbstractTopicCallback):
    topics = ["/unity_command/ground_truth/{robot_name}", "/{robot_name}/odom"]
    feature = "target"

    def callback(self, msg: Odometry, ts: rospy.Time, current_state, *args, **kwargs):
        p_pose = planar_pose(
            np.array([
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ])
        )

        p_twist = planar_twist(
            np.array(
                [
                    msg.twist.twist.linear.x,
                    msg.twist.twist.linear.y,
                    msg.twist.twist.linear.z,
                    msg.twist.twist.angular.z,
                ]
            ),
            p_pose[2],
            np.array([msg.twist.twist.angular.z]),
        )
        # Dictionaries are modified in place in python
        current_state[self.__class__.feature] = np.concatenate((p_pose, p_twist))
