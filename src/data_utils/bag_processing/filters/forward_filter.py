from genpy import Duration
from rospy import AnyMsg, Time

from data_utils.bag_processing.msg_stubs import PIDInfo
from .abstract_filter import AbstractFilter

class ForwardFilter(AbstractFilter):
    name = "forward"
    topics = [{'/{robot_name}/pid_info'}]

    def __init__(self) -> None:
        super().__init__()

        # Initialize per-bag variables
        self.end_bag()

        self.after_last_threshold_log = Duration(1)  # seconds

    @property
    def should_log(self) -> bool:
        return self.cur_state

    def callback(self, msg: PIDInfo, ts: Time, topic: str):
        self.cur_state = (
            msg.vel > 1e-6 or (ts - self.last_forward_msg) < self.after_last_threshold_log
        )

        if msg.vel > 1e-6:
            self.last_forward_msg = ts

        if not self.cur_state:
            print("ForwardFilter: Not logging")
            print(f"msg.vel: {msg.vel}")
            print(f"ts diff: {(ts - self.last_forward_msg).to_sec()}")

    def end_bag(self):
        self.cur_state = False
        self.last_forward_msg = Time()