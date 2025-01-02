from vit_prisma.vjepa_hf.configs.sae_jepa_config import JEPABaseConfig
from vit_prisma.sae.train_sae import VisionSAETrainer

# # Load HookedViT

def load_hooked_model(og_model_name = "vjepa_v1_vit_huge"):
    from vit_prisma.models.base_vit import HookedViT
    # print("Loading hooked model")
    hooked_model = HookedViT.from_pretrained(og_model_name, is_timm=False, is_clip=False, fold_ln=False) # in future, do all models
    # print("hooked model config", hooked_model.cfg)
    return hooked_model

# pr

hooked_model = load_hooked_model()


cfg = JEPABaseConfig()

print(cfg)


trainer = VisionSAETrainer(cfg)
sae = trainer.run()


