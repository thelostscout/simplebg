from abc import ABC, abstractmethod

from torch import Tensor
from torch.utils.data import TensorDataset


class BaseDataset(ABC, TensorDataset):

    def __init__(self, *tensors: Tensor):
        super().__init__(*tensors)
        self.check_channels()
        pass

    @property
    def dims(self):
        return [len(t) for t in self[0]]

    @property
    @abstractmethod
    def channels(self):
        raise NotImplementedError

    def check_channels(self):
        if len(self.dims) != len(self.channels):
            raise ValueError(f"Number of channels ({len(self.channels)}) does not match the number of Tensors "
                             f"in the features ({len(self.dims)}).")
        pass

    def __getattr__(self, item):
        try:
            return super().__getattr__(item)
        except AttributeError:
            if item in self.channels:
                return self.tensors[self.channels.index(item)]


class PeptideCCDataset(BaseDataset):
    # TODO: this returns a list with 1 tensor as output, not intuitive. Rework?
    def __init__(self, coordinates: Tensor):
        super().__init__(coordinates)

    @property
    def channels(self):
        return ["cartesian_coordinates"]


class PeptideICDataset(BaseDataset):
    def __init__(
            self,
            bonds: Tensor,
            angles: Tensor,
            torsions: Tensor,
            origin: Tensor,
            rotation: Tensor,
    ):
        super().__init__(bonds, angles, torsions, origin, rotation)

    # TODO: rework this into outputs with named tuples
    @property
    def channels(self):
        return ["bonds", "angles", "torsions", "origin", "rotation"]
