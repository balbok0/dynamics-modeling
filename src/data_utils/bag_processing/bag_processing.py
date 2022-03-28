from typing import List
from pathlib import Path
from .sequence_readers import ASyncSequenceReader, AutorallyFixedTimestepReader


def load_bag(data_folder: Path, dataset_name: str, features: List[str], robot_type: str):
    # TODO: Add parsing for skid. Bags should also be able to just subscribe to cmd_vel
    # reader = ASyncSequenceReader(features)
    reader = AutorallyFixedTimestepReader(features)

    if robot_type == "ackermann":
        for file_path in data_folder.rglob("*.bag"):
            reader.bag_extract_data(file_path)

    return reader.sequences
