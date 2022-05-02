from collections import defaultdict
from typing import Dict, List, Set

from .abstract_transform import AbstractTransform
from .autorally_ground_truth import AutorallyGroundTruth
from .autorally_state import AutorallyState
from .ground_truth_transform import GroundTruthTransform
from .input_transform import InputTransform
from .odom_transform import OdomTransform


class __TransformStore:
    __transforms: List[AbstractTransform] = [
        GroundTruthTransform(),
        InputTransform(),
        AutorallyGroundTruth(),
        AutorallyState(),
        OdomTransform(),
    ]

    @classmethod
    def get_topics_and_transforms(
        cls, features: List[str], bag_topics: Set[str], format_map: Dict[str, str]
    ) -> Dict[str, List[AbstractTransform]]:
        result = defaultdict(list)
        features_to_find = set(features)

        for transform_cls in cls.__transforms:
            if transform_cls.feature in features_to_find:
                for topic_set in transform_cls.topics:
                    topic_set = {topic.format_map(format_map) for topic in topic_set}

                    if topic_set.issubset(bag_topics):
                        # Topic exists in a bag, and feature is requested.

                        # Instantiate and append object to result
                        transform_instance = transform_cls(features)
                        for topic in topic_set:
                            result[topic].append(transform_instance)
                        features_to_find.discard(transform_cls.feature)

                        # Break looping over topic sets
                        break

        return result

    @classmethod
    def register_transform(cls, transform: AbstractTransform):
        cls.__transforms.append(transform)


# Expose public functions
get_topics_and_transforms = __TransformStore.get_topics_and_transforms
register_transform = __TransformStore.register_transform
