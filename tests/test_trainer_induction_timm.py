# import unittest
# from torchvision import datasets, transforms
# from torch.nn import CrossEntropyLoss
# from vit_prisma.models.base_vit import BaseViT  # Assuming you have a ViT model defined somewhere

# from vit_prisma.configs.InductionConfig import GlobalConfig
# from vit_prisma.training.trainer import train
# from vit_prisma.dataloaders.induction import InductionDataset
# import timm
# import torch.nn as nn


# class TestTrainingFunction(unittest.TestCase):

#         # Load MNIST dataset
#         def setUp(self):

#             transform = transforms.Compose([
#                 transforms.ToPILImage('L'),  # Convert tensor to PIL Image. Ensure your data is suitable for this.
#                 transforms.Grayscale(num_output_channels=3),  # Convert grayscale to RGB
#                 transforms.Resize((224, 224)),  # Resize to the desired dimensions
#                 transforms.ToTensor(),  # Convert back to tensor
#                 transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize the image
#             ])

#             self.train_dataset = InductionDataset('train', transform=transform)
#             self.val_dataset = InductionDataset('test', transform=transform)
    
#             self.config = GlobalConfig()

#             # Define a simple model
#             self.model = timm.create_model('vit_base_patch16_224', pretrained=True)
#             self.model.head = nn.Linear(self.model.head.in_features, 4)
        
#             # Freeze all the parameters
#             for param in self.model.parameters():
#                 param.requires_grad = False
            
#             # Unfreeze the parameters of the classification head
#             for param in self.model.head.parameters():
#                 param.requires_grad = True

#         def test_train_function(self):
#             trained_model = train(self.model, self.config, self.train_dataset, self.val_dataset)
#             self.assertIsInstance(trained_model, type(self.model))

# if __name__ == '__main__':
#     unittest.main()
