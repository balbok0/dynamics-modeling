from typing import Dict, List
import numpy as np
import h5py as h5
from pathlib import Path
from .bag_processing import bag_extract_data
from .hdf5_processing import hdf5_extract_data

SequenceData = Dict[str, np.ndarray]  # TODO: For really big datasets, this should also have a possible type of Callable[[], SequenceData]

def load_dataset(dataset_name: str, features: List[str], robot_type: str) -> List[SequenceData]:
    data_folder: Path = Path("datasets") / dataset_name

    result = []
    time_result = []

    # Numpy processing
    result.extend(load_numpy(data_folder, dataset_name, features, robot_type))

    # HDF5 Processing
    # TODO: Detect if hdf5 file is ackermann or skid. Add parsing for skid
    if robot_type == "ackermann":
        for file_path in list(data_folder.rglob("*.h5")) + list(data_folder.rglob("*.hdf5")):
            with h5.File(file_path) as f:
                result.extend(hdf5_extract_data(dataset_name, f))

    # Bag Processing
    # TODO: Add parsing for skid. Bags should also be able to just subscribe to cmd_vel
    if robot_type == "ackermann":
        for file_path in data_folder.rglob("*.bag"):
            seqs, time_seqs = bag_extract_data(dataset_name, file_path)
            result.extend(seqs)
            time_result.extend(time_seqs)

    if not result:
        raise ValueError(
            f"Dataset: {dataset_name} not found. Ensure that is a folder under \'datasets/\' directory"
        )

    # TODO: This should be based on a parameter. rn just a simple check
    if len(result) == len(time_result):
        return result, time_result
    else:
        return result, None


def load_numpy(data_folder: Path, dataset_name: str, features: List[str], robot_type: str):
    seqs = []

    # This is legacy stuff, so we can specifically do this for FIXED datasets
    if dataset_name in {"rzr_sim", "sim_data", "sim_odom_twist"}:
        # Ensure that features requested is a subset of control, state, target
        numpy_features = {"control", "state", "target"}
        if not numpy_features.issuperset(features):
            # Features is not a subset of ones that numpy can provide
            print(features)
            return []

        if dataset_name == "sim_data":
            D = 2 # cmd_vel in the form dx, dtheta
            H = 0
            # P = 3 for poses (x, y, theta)
            P = 3
        elif dataset_name == "sim_odom_twist":
            D = 2
            # This also includes odom measurements of dx, dtheta
            H = 2
            # This also includes dx, dy, dtheta
            P = 6
        elif dataset_name == "rzr_sim":
            is_ackermann = True
            D = 3 # throttle, brake, steer (multiplied by -1 if we're in reverse)
            H = 2
            P = 6

        for numpy_path in data_folder.rglob("np.txt"):
            X = np.loadtxt(numpy_path)
            N = X.shape[0]
            seq = []
            seq_no = 0
            for row in X:
                data = row[1:].reshape((1,-1))
                if row[0] != seq_no:
                    if len(seq) > 1:
                        # Extract all of the features
                        tmp = {
                                "control": seq[:, :D],
                                "state": seq[:, D:D + H],
                                "target": seq[:, -P:],
                        }
                        # Put them in an ordered pashion
                        seqs.append(
                            {
                                "time": np.ones(len(seq)),
                                **{f: tmp[f] for f in features}
                            }
                        )
                    seq = data
                    seq_no = row[0]
                else:
                    seq = np.concatenate([seq, data], 0)
    return seqs
