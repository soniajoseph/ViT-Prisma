import unittest
from torchvision import datasets, transforms
from torch.nn import CrossEntropyLoss
from vit_prisma.models.base_vit import BaseViT  # Assuming you have a ViT model defined somewhere

from vit_prisma.configs.DSpritesConfig import GlobalConfig
from vit_prisma.training.trainer import train
from vit_prisma.dataloaders.dsprites import DSpritesDataset, train_test_dataset


class TestTrainingFunction(unittest.TestCase):

        # Load MNIST dataset
        def setUp(self):

            data_path = '/content/ViT-Prisma/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
            ds = DSpritesDataset(data_path)
            self.dsprites_datasets = train_test_dataset(ds)

            self.config = GlobalConfig()

            # Define a simple model
            self.model = BaseViT(self.config)

        def test_train_function(self):
            dsprites_datasets = self.dsprites_datasets
            trained_model = train(self.model, self.config, dsprites_datasets['train'], dsprites_datasets['test'])
            self.assertIsInstance(trained_model, type(self.model))

if __name__ == '__main__':
    unittest.main()