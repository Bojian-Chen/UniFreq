import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from inference import main
from model import UniFreqConfig, UniFreqProcessor, UniFaultBackbone, TFProFMBackbone, load_model


def tiny_config(arch, num_classes=0):
    return UniFreqConfig(
        model_arch=arch, num_classes=num_classes, seq_len=32,
        unifault_patch_size=8, embed_dim=16, heads=4, depth=1,
        tf_profm_input_len=32, tf_profm_patch_len=8,
        tf_profm_num_patches=4, tf_profm_d_model=16, tf_profm_d_ffn=32,
        tf_profm_layers=1, tf_profm_heads=4, tf_profm_feature_dim=24,
        tf_profm_head_hidden=8, dropout=0.0,
    )


class AdditionalModelTest(unittest.TestCase):
    def test_round_trip_and_cli(self):
        for arch, cls, width in [('unifault', UniFaultBackbone, 16),
                                 ('TF-ProFM', TFProFMBackbone, 24)]:
            for num_classes in (0, 3):
                with self.subTest(arch=arch, num_classes=num_classes):
                    config = tiny_config(arch, num_classes)
                    source = cls(config)
                    source.train()
                    self.assertFalse(source.training)
                    self.assertFalse(any(p.requires_grad for p in source.parameters()))
                    batch = UniFreqProcessor(config)([torch.arange(24), torch.arange(48)])
                    self.assertNotIn('freq_x', batch)
                    with tempfile.TemporaryDirectory() as directory:
                        config.save_pretrained(directory)
                        state = {'module.' + k: v for k, v in source.state_dict().items()}
                        if arch == 'TF-ProFM':
                            state['module.instance_head.0.weight'] = torch.ones(2, 2)
                            state['module.cluster_head.0.weight'] = torch.ones(2, 2)
                        save_file(state, directory + '/model.safetensors')
                        loaded, _ = load_model(directory)
                        torch.manual_seed(7)
                        expected = source(batch)
                        torch.manual_seed(7)
                        torch.testing.assert_close(loaded(batch), expected)
                        self.assertEqual(tuple(expected.shape), (2, width))
                        np.save(directory + '/input.npy', np.ones((3, 48), dtype=np.float32))
                        for filename in ('a.npz', 'b.npz'):
                            main(['--model-id', directory, '--input', directory + '/input.npy',
                                  '--output', directory + '/' + filename, '--device', 'cpu',
                                  '--batch-size', '2', '--seed', '7'])
                        with np.load(directory + '/a.npz') as a, np.load(directory + '/b.npz') as b:
                            self.assertEqual(a['features'].shape, (3, width))
                            np.testing.assert_array_equal(a['features'], b['features'])
                            self.assertEqual('logits' in a, num_classes > 0)
                            if num_classes:
                                self.assertEqual(a['logits'].shape, (3, num_classes))
                        state.pop(next(k for k in state if 'weight' in k))
                        save_file(state, directory + '/model.safetensors')
                        with self.assertRaises(RuntimeError):
                            load_model(directory)
                        self.assertTrue(Path(directory, 'config.json').is_file())

    def test_unifault_base_attention_dimensions(self):
        config = UniFreqConfig(
            model_arch="unifault", embed_dim=512, heads=12, depth=16,
            seq_len=1024, unifault_patch_size=64,
        )
        model = UniFaultBackbone(config)
        attention = model.encoder.attn_layers[0].attention
        self.assertEqual(attention.query_projection.out_features, 504)
        self.assertEqual(len(model.encoder.attn_layers), 16)
        with torch.inference_mode():
            features = model(UniFreqProcessor(config)([torch.arange(1024)]))
        self.assertEqual(tuple(features.shape), (1, 512))
        self.assertTrue(torch.isfinite(features).all())

    def test_explicit_finetuning(self):
        for arch, cls in [('unifault', UniFaultBackbone), ('TF-ProFM', TFProFMBackbone)]:
            model = cls(tiny_config(arch))
            model.enable_finetuning()
            self.assertTrue(model.training)
            batch = UniFreqProcessor(model.args)([torch.arange(48)])
            model(batch).square().mean().backward()
            self.assertTrue(any(p.grad is not None for p in model.parameters()))
            model.disable_finetuning()
            model.train()
            self.assertFalse(model.training)
            self.assertFalse(any(p.requires_grad for p in model.parameters()))

    def test_preprocessing_and_validation(self):
        unifault = tiny_config('unifault')
        profm = tiny_config('TF-ProFM')
        signal = torch.arange(48, dtype=torch.float32)
        actual = UniFreqProcessor(profm)([signal])['time_x'][0, 0]
        torch.testing.assert_close(actual, signal[:32])
        actual = UniFreqProcessor(unifault)([signal])['time_x'][0, 0]
        expected = torch.nn.functional.interpolate(signal[None, None], size=32,
                                                    mode='linear', align_corners=False)[0, 0]
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            UniFreqProcessor(unifault)([torch.ones(2, 48)])
        with self.assertRaises(ValueError):
            UniFreqConfig(model_arch='TF-ProFM', tf_profm_heads=3)
        with self.assertRaises(ValueError):
            UniFreqConfig(model_arch='unifault', seq_len=33)
        config = tiny_config('unifault')
        config.input_length_mode = 'preserve'
        with self.assertRaises(ValueError):
            UniFreqProcessor(config)([signal])
