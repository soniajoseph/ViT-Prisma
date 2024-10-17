# cfg.wandb_project = cfg.model_name.replace('/', '-') + "-expansion-" + str(
# cfg.expansion_factor) + "-layer-" + str(cfg.hook_point_layer)
# cfg.unique_hash = uuid.uuid4().hex[:8]
# cfg.run_name = cfg.unique_hash + "-" + cfg.wandb_project
# wandb.init(project=cfg.wandb_project, name=cfg.run_name, entity="Stevinson")
#
# cfg.sae_path = str(MODEL_DIR / "sae/imagenet/checkpoints/ff86e305-wkcn-TinyCLIP-ViT-40M-32-Text-19M-LAION400M-expansion-16-layer-9/n_images_1300008.pt")
# model = load_model(cfg)
# sae = load_sae(cfg)
# train_data, val_data, val_data_visualize = load_dataset(cfg, visualize=True)  # TODO: I should be using test not validation data here
# q
# evaluator = Evaluator(model, val_data, cfg, visualize_data=val_data_visualize)  # TODO: Do I need the visualize data? Or can I just use the same variable
# evaluator.evaluate(sae, context=EvaluationContext.POST_TRAINING)

# This is used to create a small subset of the data in l.79 of loader.py for fast testing
# train_data = SubsetDataset(train_data, 10000)
# val_data = SubsetDataset(val_data, 1000)