# import unittest
# from torchvision import datasets, transforms
# from torch.nn import CrossEntropyLoss
# from vit_prisma.models.base_vit import BaseViT  # Assuming you have a ViT model defined somewhere

# from vit_prisma.configs.InductionConfig import GlobalConfig
# from vit_prisma.training.trainer import train
# from vit_prisma.dataloaders.induction import InductionDataset

# class TestTrainingFunction(unittest.TestCase):

#         # Load MNIST dataset
#         def setUp(self):

#             self.train_dataset = InductionDataset('train')
#             self.val_dataset = InductionDataset('test')
    
#             self.config = GlobalConfig()

#             # Define a simple model
#             self.model = BaseViT(self.config)

#         def test_train_function(self):
#             trained_model = train(self.model, self.config, self.train_dataset, self.val_dataset)
#             self.assertIsInstance(trained_model, type(self.model))

# if __name__ == '__main__':
#     unittest.main()
