from vit_prisma.configs.HookedViTConfig import HookedViTConfig
from vit_prisma.models.base_vit import HookedViT

# instantiate untrained networks to check dimensions
conf_1 = HookedViTConfig(
    n_layers=3,
    d_head=32,
    d_model=64,
    d_mlp=128,
    n_heads=2,
    patch_size=4,
)

model_1 = HookedViT(conf_1)

conf_2 = HookedViTConfig(
    n_layers=2,
    d_head=16,
    d_model=128,
    d_mlp=300,
    n_heads=8,
    patch_size=16,
)

model_2 = HookedViT(conf_2)


def test_weight_property_shapes():
    for model, conf in [(model_1, conf_1), (model_2, conf_2)]:
        # check shapes of weights
        assert model.W_Q.shape == (
            conf.n_layers,
            conf.n_heads,
            conf.d_model,
            conf.d_head,
        )
        assert model.W_K.shape == (
            conf.n_layers,
            conf.n_heads,
            conf.d_model,
            conf.d_head,
        )
        assert model.W_V.shape == (
            conf.n_layers,
            conf.n_heads,
            conf.d_model,
            conf.d_head,
        )
        assert model.W_O.shape == (
            conf.n_layers,
            conf.n_heads,
            conf.d_head,
            conf.d_model,
        )
        assert model.W_in.shape == (
            conf.n_layers,
            conf.d_model,
            conf.d_mlp,
        )
        assert model.W_out.shape == (
            conf.n_layers,
            conf.d_mlp,
            conf.d_model,
        )
        assert model.W_E.shape == (
            conf.d_model,
            conf.n_channels,
            conf.patch_size,
            conf.patch_size,
        )
        assert model.W_H.shape == (conf.d_model, conf.n_classes)

        # check shapes of biases
        assert model.b_Q.shape == (
            conf.n_layers,
            conf.n_heads,
            conf.d_head,
        )
        assert model.b_K.shape == (
            conf.n_layers,
            conf.n_heads,
            conf.d_head,
        )
        assert model.b_V.shape == (
            conf.n_layers,
            conf.n_heads,
            conf.d_head,
        )
        assert model.b_O.shape == (
            conf.n_layers,
            conf.d_model,
        )
        assert model.b_in.shape == (
            conf.n_layers,
            conf.d_mlp,
        )
        assert model.b_out.shape == (
            conf.n_layers,
            conf.d_model,
        )
        assert model.b_H.shape == (conf.n_classes,)
