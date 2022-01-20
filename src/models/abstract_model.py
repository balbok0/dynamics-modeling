from abc import ABC, abstractproperty
from typing import List

class ModelWithSpec(ABC):
    @abstractproperty
    def name(self) -> str:
        pass

    @abstractproperty
    def features(self) -> List[str]:
        pass
