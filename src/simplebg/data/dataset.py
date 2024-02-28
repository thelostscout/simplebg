from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import TensorDataset


class BaseDataset(ABC, TensorDataset):

    def __init__(self, *tensors: Tensor):
        super().__init__(*tensors)
        self.check_channels()
        pass

    @property
    def ndims(self):
        return [(len(t)) for t in self[0]]

    @property
    @abstractmethod
    def channels(self):
        raise NotImplementedError

    def check_channels(self):
        if len(self.ndims) != len(self.channels):
            raise ValueError(f"Number of channels ({len(self.channels)}) does not match the number of Tensors "
                             f"in the features ({len(self.ndims)}).")
        pass


class PeptideCCDataset(BaseDataset):
    def __init__(self, coordinates: Tensor):
        super().__init__(coordinates)

    @property
    def channels(self):
        return ["cartesian coordinates"]
