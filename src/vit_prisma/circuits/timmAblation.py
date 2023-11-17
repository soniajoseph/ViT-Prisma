import timm
import torch

class timmAblation(torch.nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained)
        self.ablation_cache = {} 

    def ablate_mlp_of_block(self, block_idx):
        block = self.model.blocks[block_idx]
        cache = {'fc1_weight': block.mlp.fc1.weight.data.clone(),
                 'fc1_bias': block.mlp.fc1.bias.data.clone(),
                 'fc2_weight': block.mlp.fc2.weight.data.clone(),
                 'fc2_bias': block.mlp.fc2.bias.data.clone()}
        self.ablation_cache[('mlp', block_idx)] = cache

        block.mlp.fc1.weight.data.fill_(0)
        block.mlp.fc1.bias.data.fill_(0)
        block.mlp.fc2.weight.data.fill_(0)
        block.mlp.fc2.bias.data.fill_(0)

    def restore_mlp_of_block(self, block_idx):
        block = self.model.blocks[block_idx]
        cache = self.ablation_cache[('mlp', block_idx)]
        if cache:
            block.mlp.fc1.weight.data = cache['fc1_weight']
            block.mlp.fc1.bias.data = cache['fc1_bias']
            block.mlp.fc2.weight.data = cache['fc2_weight']
            block.mlp.fc2.bias.data = cache['fc2_bias']
    
    def ablate_attn_head(self, block_idx, head_idx):
        block = self.model.blocks[block_idx]
        num_heads = block.attn.num_heads
        head_dim = block.attn.head_dim

        start_idx = head_idx * head_dim
        end_idx = (head_idx + 1) * head_dim

        cache = {
            'qkv_weight': block.attn.qkv.weight.data[:, start_idx:end_idx].clone(),
            'qkv_bias': block.attn.qkv.bias.data[start_idx:end_idx].clone()
        }

        block.attn.qkv.weight.data[:, start_idx:end_idx].fill_(0)
        block.attn.qkv.bias.data[start_idx:end_idx].fill_(0)
        self.ablation_cache[('attn', block_idx, head_idx)] = cache

    def restore_attn_head(self, block_idx, head_idx):
        cache = self.ablation_cache[('attn', block_idx, head_idx)]
        if cache:
            block = self.model.blocks[block_idx]
            num_heads = block.attn.num_heads
            head_dim = block.attn.head_dim

            start_idx = head_idx * head_dim
            end_idx = (head_idx + 1) * head_dim

            block.attn.qkv.weight.data[:, start_idx:end_idx] = cache['qkv_weight']
            block.attn.qkv.bias.data[start_idx:end_idx] = cache['qkv_bias']
            
    def ablate_attn_layer(self, block_idx):
        block = self.model.blocks[block_idx]
        attn = block.attn

        # Save the original weights and biases
        self.ablation_cache[('attn', block_idx)] = {
            'qkv_weight': attn.qkv.weight.data.clone(),
            'qkv_bias': attn.qkv.bias.data.clone(),
            'proj_weight': attn.proj.weight.data.clone(),
            'proj_bias': attn.proj.bias.data.clone()
        }

        # Set the weights and biases of qkv and proj to zero
        attn.qkv.weight.data.fill_(0)
        attn.qkv.bias.data.fill_(0)
        attn.proj.weight.data.fill_(0)
        attn.proj.bias.data.fill_(0)
        
    def restore_attn_layer(self, block_idx):
        if ('attn', block_idx) in self.ablation_cache:
            block = self.model.blocks[block_idx]
            attn = block.attn
            cache = self.ablation_cache[('attn', block_idx)]

            # Restore the original weights and biases
            attn.qkv.weight.data = cache['qkv_weight']
            attn.qkv.bias.data = cache['qkv_bias']
            attn.proj.weight.data = cache['proj_weight']
            attn.proj.bias.data = cache['proj_bias']


    def forward(self, x):
        return self.model(x)
    