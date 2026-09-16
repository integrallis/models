#!/usr/bin/env python3
"""Synthetic tests for applying a head-local mapper to an autoregressive cache."""

import unittest

import torch
from transformers import DynamicCache

from cross_model_kv.rope import RopeSpec, apply_rope
from evaluate_hf_mapper import _mapped_target_cache, _target_cache, map_content_cache


class HfMapperEvaluationTest(unittest.TestCase):
    def test_chunked_source_cache_translation_matches_whole_cache_translation(self):
        rope = RopeSpec(head_dim=2, rotary_dim=2, base=10_000.0)
        positions = torch.arange(3)
        content = torch.tensor(
            [
                [[[1.0, 2.0]], [[3.0, 4.0]], [[5.0, 6.0]]],
                [[[7.0, 8.0]], [[9.0, 10.0]], [[11.0, 12.0]]],
            ]
        )
        values = content + 20.0
        source_legacy = tuple(
            (
                apply_rope(content[layer], positions, rope).transpose(0, 1).unsqueeze(0),
                values[layer].transpose(0, 1).unsqueeze(0),
            )
            for layer in range(2)
        )
        source_cache = DynamicCache.from_legacy_cache(source_legacy)
        selection = torch.tensor([[1], [0]])
        weight = torch.zeros(2, 2, 2, 2)
        weight[:, :, :, :] = torch.eye(2)
        bias = torch.zeros(2, 2, 2)

        expected_keys, expected_values = map_content_cache(
            content, values, selection, weight, bias
        )
        expected = _target_cache(expected_keys, expected_values, rope)
        actual = _mapped_target_cache(
            source_cache,
            rope,
            selection,
            weight,
            bias,
            rope,
            source_token_from=0,
            token_chunk=1,
        )

        for expected_layer, actual_layer in zip(
            expected.to_legacy_cache(), actual.to_legacy_cache()
        ):
            self.assertTrue(torch.equal(expected_layer[0], actual_layer[0]))
            self.assertTrue(torch.equal(expected_layer[1], actual_layer[1]))

    def test_target_cache_appends_positioned_mapped_middle_to_native_sink(self):
        prefix = DynamicCache.from_legacy_cache(
            ((torch.tensor([[[[7.0, 8.0]]]]), torch.tensor([[[[9.0, 10.0]]]])),)
        )
        keys = torch.zeros(1, 2, 1, 2)
        values = torch.tensor([[[[20.0, 21.0]], [[30.0, 31.0]]]])
        rope = RopeSpec(head_dim=2, rotary_dim=2, base=10_000.0)

        cache = _target_cache(keys, values, rope, start_position=1, prefix_cache=prefix)
        key, value = cache.to_legacy_cache()[0]

        self.assertEqual(tuple(key.shape), (1, 1, 3, 2))
        self.assertTrue(torch.equal(key[0, 0, 0], torch.tensor([7.0, 8.0])))
        self.assertTrue(torch.equal(value[0, 0], torch.tensor([[9.0, 10.0], [20.0, 21.0], [30.0, 31.0]])))

    def test_maps_each_head_from_its_selected_source_layers(self):
        # source: layer, token, head, dimension
        source_keys = torch.tensor(
            [
                [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]],
                [[[10.0, 20.0], [30.0, 40.0]], [[50.0, 60.0], [70.0, 80.0]]],
            ]
        )
        source_values = source_keys + 100.0
        selection = torch.tensor([[1], [0]])
        # target-layer, K/V, head, input, output
        weight = torch.zeros(2, 2, 2, 2, 2)
        for layer in range(2):
            for kind in range(2):
                for head in range(2):
                    weight[layer, kind, head] = torch.eye(2) * (kind + 1)
        bias = torch.zeros(2, 2, 2, 2)

        keys, values = map_content_cache(
            source_keys, source_values, selection, weight, bias
        )

        self.assertTrue(torch.equal(keys[0], source_keys[1]))
        self.assertTrue(torch.equal(keys[1], source_keys[0]))
        self.assertTrue(torch.equal(values[0], source_values[1] * 2.0))
        self.assertTrue(torch.equal(values[1], source_values[0] * 2.0))

    def test_full_cross_head_layout_can_mix_source_heads(self):
        source_keys = torch.tensor([[[[1.0], [10.0]], [[2.0], [20.0]]]])
        source_values = source_keys + 100.0
        selection = torch.tensor([[0]])
        # target-layer, K/V, output, input -- the production Java matrix layout.
        weight = torch.zeros(1, 2, 2, 2)
        weight[0, 0] = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        weight[0, 1] = torch.tensor([[0.0, 2.0], [3.0, 0.0]])
        bias = torch.zeros(1, 2, 2)

        keys, values = map_content_cache(
            source_keys, source_values, selection, weight, bias
        )

        self.assertTrue(torch.equal(keys[0, :, 0, 0], torch.tensor([10.0, 20.0])))
        self.assertTrue(torch.equal(keys[0, :, 1, 0], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(values[0, :, 0, 0], torch.tensor([220.0, 240.0])))
        self.assertTrue(torch.equal(values[0, :, 1, 0], torch.tensor([303.0, 306.0])))


if __name__ == "__main__":
    unittest.main()
