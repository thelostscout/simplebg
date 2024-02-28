from bgmol.datasets import Ala2TSF300
from torch import Tensor

from .dataset import PeptideCCDataset


def main():
    ala = Ala2TSF300(download=False, read=True)
    t =Tensor(ala.coordinates).flatten(1)

    return PeptideCCDataset(t)
