from simplebg.data.dataset import PeptideCCDataset, PeptideICDataset
import unittest
from abc import ABC, abstractmethod
import torch
from torch import Tensor


# @unittest.skip("Abstract class")
class AbstractDatasetTestCase(unittest.TestCase, ABC):
    @abstractmethod
    def test_ndims(self):
        raise NotImplementedError

    @abstractmethod
    def test_channels(self):
        raise NotImplementedError

    @abstractmethod
    def test___getattr__(self):
        raise NotImplementedError


class PeptideCCDatasetTestCase(unittest.TestCase):
    def test_ndims(self):
        t = torch.rand(3, 4)
        dataset = PeptideCCDataset(t)
        self.assertEqual(dataset.dims, [4])

    def test_channels(self):
        t = torch.rand(1, 1)
        dataset = PeptideCCDataset(t)
        self.assertEqual(dataset.channels, ["cartesian_coordinates"])

    def test___getattr__(self):
        t = torch.rand(3, 4)
        dataset = PeptideCCDataset(t)
        self.assertTrue(torch.equal(dataset.cartesian_coordinates, t))


class PeptideICDatasetTestCase(unittest.TestCase):

    def test_ndims(self):
        t = torch.rand(5, 4, 3)
        dataset = PeptideICDataset(*t)
        self.assertEqual(dataset.dims, [3, 3, 3, 3, 3])

    def test_channels(self):
        t = torch.rand(5, 1, 1)
        dataset = PeptideICDataset(*t)
        self.assertEqual(dataset.channels, ["bonds", "angles", "torsions", "origin", "rotation"])

    def test___getattr__(self):
        t = torch.rand(5, 3, 4)
        dataset = PeptideICDataset(*t)
        self.assertTrue(torch.equal(dataset.bonds, t[0]))
        self.assertTrue(torch.equal(dataset.angles, t[1]))
        self.assertTrue(torch.equal(dataset.torsions, t[2]))
        self.assertTrue(torch.equal(dataset.origin, t[3]))
        self.assertTrue(torch.equal(dataset.rotation, t[4]))
