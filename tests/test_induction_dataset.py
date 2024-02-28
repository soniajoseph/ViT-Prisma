import unittest
from vit_prisma.dataloaders.induction import InductionDataset
import torch
import random

SAMPLE_SIZE = 100  # Adjust based on your needs

class TestInductionDataset(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        dir_path = '../data/induction'
        cls.train_dataset = InductionDataset('train', dir_path)
        cls.test_dataset = InductionDataset('test', dir_path)

    def test_dataset_lengths(self):
        # Ensure datasets are not empty
        self.assertTrue(all(len(dataset) > 0 for dataset in [self.train_dataset, self.test_dataset]))

    def _get_sampled_class_counts(self, dataset):
        sampled_indices = random.sample(range(len(dataset)), SAMPLE_SIZE)
        labels = [dataset[i][1] for i in sampled_indices]
        return {label: labels.count(label) for label in set(labels)}

    def test_balanced_classes(self):
        for dataset in [self.train_dataset, self.test_dataset]:
            class_counts = self._get_sampled_class_counts(dataset)
            print("Class counts:", class_counts)

    def test_get_item(self):
        for dataset in [self.train_dataset, self.test_dataset]:
            image, label = dataset[0]
            self.assertEqual(image.shape[0], 1)
            self.assertIsInstance(image, torch.Tensor)

if __name__ == '__main__':
    unittest.main()
