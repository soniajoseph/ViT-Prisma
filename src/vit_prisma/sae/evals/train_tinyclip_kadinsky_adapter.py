import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPVisionModelWithProjection, UNet2DConditionModel
from diffusers import KandinskyV22PriorPipeline, KandinskyV22Pipeline
import torchvision.transforms as transforms
from torchvision.datasets import ImageNet
from tqdm import tqdm


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Assuming you have a TinyClip model implementation

# Updated Adapter Architecture
class EmbeddingAdapter(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=1280):
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.activation1 = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation2 = nn.ReLU()
        self.linear3 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.dropout(x)
        x = self.linear3(x)
        return x

# Activations Store
class DualEmbedder:
    def __init__(self, tinyclip_model, kandinsky_model):
        self.tinyclip_model = tinyclip_model
        self.kandinsky_model = kandinsky_model
        
    def get_embeddings(self, images):
        with torch.no_grad(): 
            tinyclip_embeddings = self.tinyclip_model.encode_image(images)
            kandinsky_embeddings = self.kandinsky_model.image_encoder(images).image_embeds
        return tinyclip_embeddings, kandinsky_embeddings

# Dataset
class ImageDataset(Dataset):
    def __init__(self, imagenet_data, dual_embedder, tinyclip_transform, kandinsky_transform):
        self.imagenet_data = imagenet_data
        self.dual_embedder = dual_embedder
        self.tinyclip_transform = tinyclip_transform
        self.kandinsky_transform = kandinsky_transform # Do I need these separate transforms, or do the models already come with them? 
        
    def __len__(self):
        return len(self.imagenet_data)
    
    def __getitem__(self, idx):
        image, _ = self.imagenet_data[idx]
        
        # Apply different transforms for TinyClip and Kandinsky
        tinyclip_image = self.tinyclip_transform(image)
        kandinsky_image = self.kandinsky_transform(image)
        
        # Get embeddings
        tinyclip_embed, kandinsky_embed = self.dual_embedder.get_embeddings(tinyclip_image.unsqueeze(0), kandinsky_image.unsqueeze(0))
        
        return tinyclip_embed.squeeze(0), kandinsky_embed.squeeze(0)
    
# Training function
def train_adapter(adapter, dataloader, num_epochs=10):
    optimizer = optim.Adam(adapter.parameters(), lr=1e-4)
    criterion = nn.MSELoss()
    
    for epoch in range(num_epochs):
        total_loss = 0
        for tinyclip_embed, kandinsky_embed in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            tinyclip_embed = tinyclip_embed.to(DEVICE)
            kandinsky_embed = kandinsky_embed.to(DEVICE)
            
            optimizer.zero_grad()
            output = adapter(tinyclip_embed)
            loss = criterion(output, kandinsky_embed)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(dataloader)}")

# Kandinsky loading function
def load_kandinsky(cache_dir):
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        'kandinsky-community/kandinsky-2-2-prior',
        subfolder='image_encoder'
        cache_dir=cache_dir,
    ).half().to(DEVICE)

    unet = UNet2DConditionModel.from_pretrained(
        'kandinsky-community/kandinsky-2-2-decoder',
        subfolder='unet',
        cache_dir=cache_dir,
    ).half().to(DEVICE)

    prior = KandinskyV22PriorPipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-prior',
        image_encoder=image_encoder,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    ).to(DEVICE)

    decoder = KandinskyV22Pipeline.from_pretrained(
        'kandinsky-community/kandinsky-2-2-decoder',
        unet=unet,
        torch_dtype=torch.float16,
        cache_dir=cache_dir,
    ).to(DEVICE)

    zero_embed = prior.get_zero_embed()

    return prior, decoder, zero_embed

# Function to get TinyClip embedding
def get_tinyclip_embedding(prompt):
    # Implement your TinyClip embedding logic here
    # This is a placeholder, replace with actual implementation
    return torch.randn(512).to(DEVICE)

# Kandinsky loading function with adapter
def load_kandinsky_with_adapter(adapter):
    prior, decoder, zero_embed = load_kandinsky()
    
    # Wrap the prior's encode_prompt function
    original_encode_prompt = prior.encode_prompt
    
    def encode_prompt_with_adapter(prompt, num_images_per_prompt, do_classifier_free_guidance):
        tinyclip_embed = get_tinyclip_embedding(prompt)
        adapted_embed = adapter(tinyclip_embed.unsqueeze(0)).squeeze(0)
        return original_encode_prompt(adapted_embed, num_images_per_prompt, do_classifier_free_guidance)
    
    prior.encode_prompt = encode_prompt_with_adapter
    
    return prior, decoder, zero_embed

# Main script
if __name__ == "__main__":
    # Define image transformations
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load ImageNet dataset
    imagenet_path = 'path/to/imagenet'  # Replace with your ImageNet path
    imagenet_data = ImageNet(root=imagenet_path, split='train', transform=transform)

    # Initialize TinyClip and Kandinsky models
    tinyclip_model = TinyClip().to(DEVICE)
    tinyclip_model.eval()
    
    prior, _, _ = load_kandinsky()
    kandinsky_model = prior.image_encoder
    kandinsky_model.eval()

    # Create activations store
    activations_store = DualEmbedder(tinyclip_model, kandinsky_model)

    # Create dataset and dataloader
    dataset = ImageDataset(imagenet_data, activations_store)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Initialize adapter with larger hidden layer
    adapter = EmbeddingAdapter(input_dim=512, hidden_dim=2048, output_dim=1280).to(DEVICE)

    # Train adapter
    train_adapter(adapter, dataloader)

    # Save adapter
    torch.save(adapter.state_dict(), 'tinyclip_to_kandinsky_adapter.pth')

    # # Load adapter (for demonstration)
    # adapter = EmbeddingAdapter(input_dim=512, hidden_dim=2048, output_dim=1280)
    # adapter.load_state_dict(torch.load('tinyclip_to_kandinsky_adapter.pth'))
    # adapter.to(DEVICE)

    # # Load Kandinsky with adapter
    # prior, decoder, zero_embed = load_kandinsky_with_adapter(adapter)

    # # Example usage
    # prompt = "A beautiful landscape"
    # images = prior(prompt=prompt, num_inference_steps=25, num_images_per_prompt=1)
    # image = decoder(image_embeds=images.image_embeds, prompt=prompt, num_inference_steps=50).images[0]
    # image.save("generated_image.png")

    # print("Image generated and saved as 'generated_image.png'")