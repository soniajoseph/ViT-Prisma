# import unittest
# from torchvision import datasets, transforms
# from torch.nn import CrossEntropyLoss
# from vit_prisma.models.base_vit import BaseViT  # Assuming you have a ViT model defined somewhere

# from vit_prisma.configs.MNISTConfig import GlobalConfig
# from vit_prisma.training.trainer import train

# class TestTrainingFunction(unittest.TestCase):

#         # Load MNIST dataset
#         def setUp(self):

#             data_root_dir = '../../data'
#             transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
#             self.train_dataset = datasets.MNIST(root=data_root_dir, train=True, transform=transform, download=True)
#             self.val_dataset = datasets.MNIST(root=data_root_dir, train=False, transform=transform, download=True)

#             self.config = GlobalConfig()

#             # Define a simple model
#             self.model_function = BaseViT

#         def test_train_function(self):
#             self.config.logging.wandb_team_name = 'perceptual-alignment'
#             self.config.training.early_stopping = False

#             trained_model = train(self.model_function, self.config, self.train_dataset, self.val_dataset)
#             self.assertIsInstance(trained_model, type(self.model))

# if __name__ == '__main__':
#     unittest.main()
