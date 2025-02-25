
CONFIG = {
    "model_type": "vit",
    "hidden_size": 1024,  # embed_dim from vit_large
    "num_hidden_layers": 24,  # depth from vit_large
    "num_attention_heads": 16,  # num_heads from vit_large
    "intermediate_size": 4096,  # hidden_size * mlp_ratio (4)
    "hidden_act": "gelu",
    "hidden_dropout_prob": 0.0,  # drop_rate
    "attention_probs_dropout_prob": 0.0,  # attn_drop_rate
    "initializer_range": 0.02,  # init_std
    "layer_norm_eps": 1e-6,
    "patch_size": 16,
    "image_size": 224,
    "num_channels": 3,
    "qkv_bias": True,
    "num_frames": 16,  # from config frames_per_clip
    "tubelet_size": 2,  # from config
    "use_mean_pooling": True,
    "num_classes": 174,  # from config
    "pos_embed_type": "learnable",
    "num_segments": 2,  # from config
    "num_views_per_segment": 3,  # from config
}
