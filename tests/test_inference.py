import tempfile
import unittest

import numpy as np
import torch
from safetensors.torch import save_file

from model import PHMFDConfig
from model import PHMFDProcessor
from model import TimeFrequencyBackbone
from model import load_model


def _tiny_config():
    return PHMFDConfig(
        embed_dim=16,
        heads=4,
        depth=1,
        dropout=0.0,
        window_seconds=0.1,
        time_patch_seconds=0.025,
        time_transformer_layers=1,
        freq_max_bins=16,
        freq_transformer_layers=1,
    )


class InferenceTest(unittest.TestCase):
    def test_variable_length_forward(self):
        config = _tiny_config()
        processor = PHMFDProcessor(config)
        batch = processor(
            [
                np.linspace(-1.0, 1.0, 64, dtype=np.float32),
                np.linspace(1.0, -1.0, 80, dtype=np.float32),
            ]
        )
        model = TimeFrequencyBackbone(config).eval()
        with torch.inference_mode():
            time_features, freq_features = model.branch_features(batch)
            features = model(batch)

        self.assertEqual(tuple(time_features.shape), (2, 16))
        self.assertEqual(tuple(freq_features.shape), (2, 16))
        self.assertEqual(tuple(features.shape), (2, 32))
        self.assertFalse(any(parameter.requires_grad for parameter in model.parameters()))
        model.train()
        self.assertFalse(model.training)

    def test_local_hub_round_trip(self):
        torch.manual_seed(7)
        config = _tiny_config()
        source = TimeFrequencyBackbone(config).eval()
        processor = PHMFDProcessor(config)
        batch = processor([np.arange(64, dtype=np.float32)])

        with tempfile.TemporaryDirectory() as directory:
            config.save_pretrained(directory)
            save_file(
                source.state_dict(),
                directory + "/model.safetensors",
            )
            loaded, loaded_config = load_model(directory)

            with torch.inference_mode():
                expected = source(batch)
                actual = loaded(batch)

        self.assertEqual(loaded_config.model_type, "phmfd-timefreq")
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
