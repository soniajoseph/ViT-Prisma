from vit_prisma.vjepa_hf.modeling_vjepa import VJEPAConfig

CONFIGS = {
    "v1": {
        "vit_h": VJEPAConfig(
            model_name="vit_huge",
            uniform_power=True,
            hidden_size=1280,
            num_attention_heads=16,
            num_hidden_layers=32,
            layer_norm_eps=1e-6,
            use_sdpa=True,
        ),
        "vit_h_384": VJEPAConfig(
            model_name="vit_huge",
            crop_size=384,
            uniform_power=True,
            hidden_size=1280,
            num_attention_heads=16,
            num_hidden_layers=32,
            layer_norm_eps=1e-6,
            use_sdpa=True,
        ),
    },
    "v1.5": {
        "vit_g_256": VJEPAConfig(
            model_name="vit_giant_xformers_rope",
            crop_size=256,
            uniform_power=True,
            hidden_size=1408,
            num_attention_heads=22,
            num_hidden_layers=40,
            use_rope=True,
            mlp_ratio=48 / 11,
            use_sdpa=True,
        )
    },
}
