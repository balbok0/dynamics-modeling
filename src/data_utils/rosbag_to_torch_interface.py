import rosbag_to_torch
from .numpy_set import NumpyDataset

datasets = {
    "torch_lookahead": rosbag_to_torch.LookaheadSequenceDataset,
    "torch_lookahead_diff": rosbag_to_torch.LookaheadDiffSequenceDataset,
    "numpy": NumpyDataset,
}

filters = {
    "forward": rosbag_to_torch.filters.ForwardFilter,
    "pid_info": rosbag_to_torch.filters.PIDInfoFilter,
}

readers = {
    "async": rosbag_to_torch.readers.ASyncSequenceReader,
    "fixed_interval": rosbag_to_torch.readers.FixedIntervalReader,
}
