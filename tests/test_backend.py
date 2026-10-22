"""CPU-only checks of shared preprocessing and stateful inference contracts.

Uses a stochastic stand-in only to test seed handling/cache behavior, not model
quality. Actual pretrained GPU integration is validated separately.
"""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dehaze import models


class PreprocessingTests(unittest.TestCase):
    def test_aspect_resize_and_no_upscale(self):
        image = np.full((503, 999, 3), 80, np.uint8)
        result, meta = models.prepare_image(image)
        self.assertEqual(result.shape, (258, 512, 3))
        self.assertEqual(meta["original_size"], [999, 503])
        small = np.full((7, 11, 3), 90, np.uint8)
        result, _ = models.prepare_image(small)
        np.testing.assert_array_equal(result, small)
        result[0, 0] = 0
        self.assertEqual(int(small[0, 0, 0]), 90)

    def test_padding_preserves_all_valid_pixels(self):
        image = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
        padded, meta = models._pad_image(image)
        self.assertEqual(padded.shape, (64, 64, 3))
        np.testing.assert_array_equal(padded[:5, :7], image)
        self.assertEqual(meta["padding"]["bottom"], 59)
        tiny, _ = models._pad_image(np.ones((1, 1, 3), np.uint8))
        self.assertEqual(tiny.shape, (64, 64, 3))
        self.assertTrue(np.all(tiny == 1))

    def test_input_contract(self):
        for bad in [np.zeros((2, 3, 3), np.float32), [[1, 2]]]:
            with self.assertRaises(TypeError):
                models.prepare_image(bad)
        for bad in [np.zeros((2, 3), np.uint8), np.zeros((0, 3, 3), np.uint8)]:
            with self.assertRaises(ValueError):
                models.prepare_image(bad)
        with self.assertRaises(ValueError):
            models.prepare_image(np.zeros((2, 3, 3), np.uint8), 0)


class InferenceStateTests(unittest.TestCase):
    def test_fixed_seed_warmup_and_single_shared_cache(self):
        load_calls = []
        def load_stub(root, method, device):
            load_calls.append(method)
            return models._Bundle(method, (), "float32", 0,
                                  lambda x: torch.rand_like(x) * 2 - 1)
        with patch.object(models, "_load_bundle", load_stub), patch.object(torch.cuda, "is_available", return_value=False):
            manager = models.ModelManager(device="cpu")
            manager.close()
            image = np.full((77, 101, 3), 120, np.uint8)
            original_rng = torch.random.get_rng_state().clone()
            first, meta1 = manager.restore(image, "PixRestore-S", seed=17)
            second, meta2 = manager.restore(image, "PixRestore-S", seed=17)
            third, _ = manager.restore(image, "PixRestore-S", seed=18)
            np.testing.assert_array_equal(first, second)
            self.assertFalse(np.array_equal(first, third))
            self.assertTrue(torch.equal(original_rng, torch.random.get_rng_state()))
            self.assertEqual(first.shape, image.shape)
            self.assertEqual(first.dtype, np.uint8)
            self.assertEqual(meta1["warmup_runs"], 1)
            self.assertEqual(meta2["warmup_runs"], 0)
            self.assertFalse(meta1["model_cache_hit"])
            self.assertTrue(meta2["model_cache_hit"])
            self.assertIsNone(meta1["gpu_ms"])
            other = models.ModelManager(device="cpu")
            other.restore(image, "PGL-Net-T", seed=17)
            self.assertEqual(load_calls, ["PixRestore-S", "PGL-Net-T"])
            manager.restore(image, "PixRestore-S", seed=17)
            self.assertEqual(load_calls, ["PixRestore-S", "PGL-Net-T", "PixRestore-S"])
            manager.close()


if __name__ == "__main__":
    unittest.main()
