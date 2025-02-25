import torch
import copy
import unittest

# Import the model
from jepa.models.attentive_pooler import AttentiveClassifier  # Ensure this is the correct import path

class TestModelLoading(unittest.TestCase):
    def setUp(self):
        """Set up model and load state dictionary"""
        self.model = classifier = AttentiveClassifier(embed_dim=1024,num_heads = 16, depth=1, num_classes=1000, complete_block=True) # Instantiate the model
        self.state_dict_path =  '/network/scratch/s/sonia.joseph/jepa_models/github_models/vit-l-16/probes/in1k-probe.pth.tar.1'


        # Clone model state before loading
        self.model_before = copy.deepcopy(self.model.state_dict())

        # Load state dictionary
        self.state_dict = torch.load(self.state_dict_path)['classifier']
        # get rid of 'module' prefix
        self.state_dict = {k.replace('module.', ''): v for k, v in self.state_dict.items()}
        self.model.load_state_dict(self.state_dict)

        # Clone model state after loading
        self.model_after = copy.deepcopy(self.model.state_dict())

    def test_weights_change_after_loading(self):
        """Ensure that at least some parameters change after loading"""
        changed = any(
            not torch.equal(self.model_before[k], self.model_after[k])
            for k in self.model_before.keys()
        )
        self.assertTrue(changed, "Weights did not change after loading the state dict.")

    def test_all_expected_keys_loaded(self):
        """Ensure that all keys from the state dictionary exist in the model"""
        model_keys = set(self.model.state_dict().keys())
        state_dict_keys = set(self.state_dict.keys())

        missing_keys = state_dict_keys - model_keys
        unexpected_keys = model_keys - state_dict_keys

        self.assertTrue(
            not missing_keys, f"Missing keys in model: {missing_keys}"
        )
        self.assertTrue(
            not unexpected_keys, f"Unexpected keys in model: {unexpected_keys}"
        )

    def test_weight_statistics(self):
        """Check if key weight statistics change after loading"""
        params_to_check = [
            "linear.weight",
            "pooler.query_tokens",
            "pooler.cross_attention_block.xattn.q.weight",
        ]

        for param in params_to_check:
            if param in self.model_before and param in self.model_after:
                mean_before = self.model_before[param].mean().item()
                mean_after = self.model_after[param].mean().item()
                self.assertNotAlmostEqual(
                    mean_before, mean_after, places=6,
                    msg=f"Mean of {param} did not change after loading."
                )

if __name__ == "__main__":
    unittest.main()
