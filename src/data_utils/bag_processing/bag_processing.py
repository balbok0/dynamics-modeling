from typing import Dict, List, Union
import rosbag
from pathlib import Path
import numpy as np
from .topic_callbacks import get_topics_and_callbacks
from tqdm import tqdm


def verify_sequence(sequence: Dict[str, List], features: List[str]):
    # Verify that:
    #   1. All of requested features (and "time") are present in a sequence
    #   2, Lengths of sequence agree for all key-words
    if not (set(features) | {"time"}).issubset(sequence.keys()):
        return False

    len_seq = len(sequence["time"])
    return all([len(sequence[k]) == len_seq for k in features])


def bag_extract_data(dataset_name: str, features: List[str], bag_file_path: Union[str, Path]):
    bag_file_path = Path(bag_file_path)
    bag = rosbag.Bag(bag_file_path)
    topics = bag.get_type_and_topic_info().topics

    # Get robot name. It should be the most often occuring topmost key
    topic_parents, topic_parents_counts =  np.unique([x.split("/", 2)[1] for x in topics.keys()], return_counts=True)
    robot_name = topic_parents[np.argmax(topic_parents_counts)]

    topics_with_callbacks = get_topics_and_callbacks(
        features, set(topics.keys()), {"robot_name": robot_name}
    )

    # Result holds a list of dictionaries. One for each sequence.
    result = []

    # Current state is modified in-place
    current_state = {}
    for topic, msg, ts in tqdm(
        bag.read_messages(topics=topics_with_callbacks.keys()),
        total=sum([topics[k].message_count for k in topics_with_callbacks.keys()]),
        desc=f"Extracting from bag: {bag_file_path.name}"
    ):
        for callback in topics_with_callbacks[topic]:
            finished_sequence = callback.callback(msg, ts, current_state)

            # End of current sequence
            if finished_sequence is not None:
                if not verify_sequence(finished_sequence, features):
                    raise ValueError(
                        f"Received invalid sequence. Features: {features}, but keys of sequence are: {finished_sequence.keys()}"
                        "\nIt these match it probably is an issue with shapes of these features."
                    )
                result.append(finished_sequence)

    for callback_arr in topics_with_callbacks.values():
        for callback in callback_arr:
            finished_sequence = callback.end_bag()

            # End of current sequence
            if finished_sequence is not None:
                if not verify_sequence(finished_sequence, features):
                    raise ValueError(
                        f"Received invalid sequence. Features: {features}, but keys of sequence are: {finished_sequence.keys()}"
                        "\nIt these match it probably is an issue with shapes of these features."
                    )
                result.append(finished_sequence)

    print(f"Result len: {len(result)}")
    return result


def load_bag(data_folder: Path, dataset_name: str, features: List[str], robot_type: str):
    # TODO: Add parsing for skid. Bags should also be able to just subscribe to cmd_vel
    result = []
    if robot_type == "ackermann":
        for file_path in data_folder.rglob("*.bag"):
            seqs = bag_extract_data(dataset_name, features, file_path)
            result.extend(seqs)
    return result
