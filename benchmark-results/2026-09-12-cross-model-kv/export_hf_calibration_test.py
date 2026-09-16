import unittest

import torch

from export_hf_calibration import canonical_layers


class FakeDynamicCache:
    def __init__(self, layers):
        self._layers = layers

    def to_legacy_cache(self):
        return self._layers


class CanonicalLayersTest(unittest.TestCase):
    def test_converts_dynamic_cache_to_token_kv_head_dimension_layout(self):
        key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
        value = key + 100

        layers = canonical_layers(FakeDynamicCache(((key, value),)))

        self.assertEqual((3, 2, 2, 4), tuple(layers[0].shape))
        torch.testing.assert_close(layers[0][:, 0], key[0].transpose(0, 1))
        torch.testing.assert_close(layers[0][:, 1], value[0].transpose(0, 1))

    def test_rejects_a_cache_with_more_than_one_batch(self):
        key = torch.zeros((2, 2, 3, 4))
        value = torch.zeros_like(key)

        with self.assertRaisesRegex(ValueError, "batch size 1"):
            canonical_layers(((key, value),))


if __name__ == "__main__":
    unittest.main()
