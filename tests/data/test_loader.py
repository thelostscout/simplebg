import random
import unittest
import torch
from simplebg.data.loader import split_dataset, SplitHParams


# class PeptideLoaderTestCase(unittest.TestCase):
# this is more complex because it requires test molecules

class SplitDatasetTestCase(unittest.TestCase):
    def test_DataSplitHParams(self):
        hparams = {"train_split": 0.6, "val_split": 0.2, "test_split": 0.1}
        flag = False
        try:
            hparams = SplitHParams(**hparams)
        except ValueError as e:
            print(e)
            flag = True
        self.assertTrue(flag)

    def test_split_dataset(self):
        t = torch.arange(10)
        dataset = torch.utils.data.TensorDataset(t)
        hparams = {"train_split": 0.6, "val_split": 0.2, "test_split": 0.2, "seed": 0}
        hparams = SplitHParams(**hparams)
        random.seed(hparams.seed)
        # insert here to test if random.seed and the rng in split_dataset are independent of each other
        train_data, val_data, test_data = split_dataset(dataset, hparams)
        idx = list(range(10))
        random.shuffle(idx)
        self.assertIsInstance(train_data, torch.utils.data.Subset)
        self.assertEqual(len(train_data), 6)
        self.assertListEqual(idx[:6], train_data.indices)
