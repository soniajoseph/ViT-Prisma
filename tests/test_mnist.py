import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

# Assuming you have the Config and Transformer model defined as before
from vit_prisma.configs.MNISTConfig import Config
from vit_prisma.models.base_vit import BaseViT as Transformer

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
data_root = '/home/mila/s/sonia.joseph/ViT-Planetarium/data'
train_dataset = torchvision.datasets.MNIST(root=data_root, train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.MNIST(root=data_root, train=False, transform=transform, download=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize model, criterion, and optimizer
config = Config()
model = Transformer(config)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005)

model.cuda()

# Train for a few epochs
num_epochs = 5

def compute_accuracy(loader, model):
    total_correct = 0
    total_samples = 0
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.cuda(), labels.cuda()
            outputs = model(images)
            _, predicted = outputs.max(1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)
    return 100.0 * total_correct / total_samples

for epoch in range(num_epochs):
    idx = 0
    model.train()
    for images, labels in tqdm(train_loader):
        # Flatten MNIST images into a 784-length vector
        # images = images.view(images.size(0), -1)

        images, labels = images.cuda(), labels.cuda()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if idx % 200 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        idx += 1
        

    test_accuracy = compute_accuracy(test_loader, model)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Test Accuracy: {test_accuracy:.2f}%")

print("Training complete!")