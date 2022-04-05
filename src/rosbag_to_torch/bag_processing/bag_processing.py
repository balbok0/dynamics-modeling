from typing import List
from pathlib import Path
from .sequence_readers.abstract_sequence_reader import AbstractSequenceReader, Sequences


def load_bags(data_folder: Path, reader: AbstractSequenceReader) -> Sequences:
    for file_path in data_folder.rglob("*.bag"):
        reader.extract_bag_data(file_path)

    return reader.sequences
