#!/usr/bin/env python3
"""Synthetic correctness tests for the audited head-local mapper fitter."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from fit_head_local import (
    cachebridge_selection_protocol,
    center_normal_equations_in_place,
    centered_ridge,
    combine_selection_scores,
    effective_sample_size,
    fit_kind,
    shrink_weights,
    validate_attention_manifest,
)


class HeadLocalFitterTest(unittest.TestCase):
    def test_centered_normal_equations_reuse_the_moment_storage(self):
        xx = torch.tensor(
            [[[[20.0, 8.0], [8.0, 10.0]]]], dtype=torch.float32
        )
        xy = torch.tensor(
            [[[[12.0], [7.0]]]], dtype=torch.float32
        )
        mean_x = torch.tensor([[[2.0, 1.0]]])
        mean_y = torch.tensor([[[1.5]]])
        total_weight = torch.tensor([[4.0]], dtype=torch.float64)
        xx_storage = xx.data_ptr()
        xy_storage = xy.data_ptr()

        covariance, cross = center_normal_equations_in_place(
            xx, xy, mean_x, mean_y, total_weight, regularization=0.25
        )

        self.assertEqual(covariance.data_ptr(), xx_storage)
        self.assertEqual(cross.data_ptr(), xy_storage)
        self.assertTrue(
            torch.allclose(
                covariance,
                torch.tensor([[[[1.25, 0.0], [0.0, 1.75]]]]),
            )
        )
        self.assertTrue(torch.allclose(cross, torch.tensor([[[[0.0], [0.25]]]])))

    def test_cachebridge_selection_uses_all_calibration_rows_as_fit_and_score(self):
        fit_rows, score_rows, regularization = cachebridge_selection_protocol(
            calibration_rows=(0, 128_000),
            rows_per_sequence=256,
            calibration_sequences=500,
        )

        self.assertEqual(fit_rows, (0, 128_000))
        self.assertEqual(score_rows, fit_rows)
        self.assertEqual(regularization, 0.0)

    def test_cachebridge_selection_refuses_partial_calibration(self):
        with self.assertRaisesRegex(ValueError, "all 500 calibration sequences"):
            cachebridge_selection_protocol(
                calibration_rows=(0, 128_000),
                rows_per_sequence=256,
                calibration_sequences=100,
            )

    def test_attention_weighting_rejects_pre_correction_manifests(self):
        manifest = {
            "schemaVersion": 1,
            "method": "same-token receiver-attention sensitivities",
            "revision": "target-revision",
            "shape": [100, 2, 1, 2],
        }

        with self.assertRaisesRegex(ValueError, "schema version 2"):
            validate_attention_manifest(
                manifest,
                expected_model="target-model",
                expected_revision="target-revision",
                expected_sequence_manifest_sha256="sequence-sha",
                expected_sequence_ids=["seq-0", "seq-1"],
                expected_sampled_positions=[0, 4],
                target_layers=2,
                kv_heads=1,
            )

    def test_attention_weighting_accepts_equal_mass_boundary_manifest(self):
        manifest = {
            "schemaVersion": 2,
            "method": (
                "CacheBridge first-future-query receiver-attention sensitivities "
                "with per-boundary unit-mass aggregation"
            ),
            "revision": "target-revision",
            "model": "target-model",
            "sequenceManifestSha256": "sequence-sha",
            "eligiblePrefix": "positions strictly before each query boundary",
            "boundaryAggregation": (
                "normalize each layer, KV head, and K/V component to unit mass at "
                "every boundary, then sum boundaries with equal mass"
            ),
            "sequenceIds": ["seq-0", "seq-1"],
            "sampledPositions": [0, 4],
            "shape": [4, 2, 1, 2],
            "layout": "observation,target-layer,kv-head,key-or-value",
            "dtype": "float32",
            "reference": {
                "paper": "https://arxiv.org/abs/2609.00891",
                "version": "v1",
                "equations": [15, 16, 17, 18, 19],
            },
        }

        validate_attention_manifest(
            manifest,
            expected_model="target-model",
            expected_revision="target-revision",
            expected_sequence_manifest_sha256="sequence-sha",
            expected_sequence_ids=["seq-0", "seq-1"],
            expected_sampled_positions=[0, 4],
            target_layers=2,
            kv_heads=1,
        )

    def test_attention_weighting_rejects_sequence_reordering(self):
        manifest = {
            "schemaVersion": 2,
            "method": (
                "CacheBridge first-future-query receiver-attention sensitivities "
                "with per-boundary unit-mass aggregation"
            ),
            "revision": "target-revision",
            "model": "target-model",
            "sequenceManifestSha256": "sequence-sha",
            "sequenceIds": ["seq-1", "seq-0"],
            "sampledPositions": [0, 4],
            "eligiblePrefix": "positions strictly before each query boundary",
            "boundaryAggregation": (
                "normalize each layer, KV head, and K/V component to unit mass at "
                "every boundary, then sum boundaries with equal mass"
            ),
            "shape": [4, 2, 1, 2],
            "layout": "observation,target-layer,kv-head,key-or-value",
            "dtype": "float32",
            "reference": {
                "paper": "https://arxiv.org/abs/2609.00891",
                "version": "v1",
                "equations": [15, 16, 17, 18, 19],
            },
        }

        with self.assertRaisesRegex(ValueError, "sequence order"):
            validate_attention_manifest(
                manifest,
                expected_model="target-model",
                expected_revision="target-revision",
                expected_sequence_manifest_sha256="sequence-sha",
                expected_sequence_ids=["seq-0", "seq-1"],
                expected_sampled_positions=[0, 4],
                target_layers=2,
                kv_heads=1,
            )

    def test_common_layer_selection_uses_mean_key_and_value_score(self):
        scores = {
            "keys": torch.tensor([[0.9, 0.8, 0.1], [0.1, 0.7, 0.8]]),
            "values": torch.tensor([[0.1, 0.7, 0.95], [0.9, 0.6, 0.2]]),
        }

        selection, combined = combine_selection_scores(scores, top_k=2)

        self.assertEqual(selection.tolist(), [[1, 2], [1, 0]])
        self.assertTrue(torch.allclose(combined, torch.tensor([[0.5, 0.75, 0.525], [0.5, 0.65, 0.5]])))

    def test_centered_ridge_recovers_an_affine_map(self):
        generator = torch.Generator().manual_seed(7)
        x = torch.randn(512, 4, generator=generator)
        expected_weight = torch.tensor(
            [[1.2, -0.4], [0.1, 0.7], [-0.8, 0.2], [0.3, 1.1]]
        )
        expected_bias = torch.tensor([0.35, -0.6])
        y = x @ expected_weight + expected_bias

        weight, bias = centered_ridge(x, y, regularization=1.0e-8)

        self.assertTrue(torch.allclose(weight, expected_weight, atol=2.0e-5))
        self.assertTrue(torch.allclose(bias, expected_bias, atol=2.0e-5))

    def test_zero_weight_excludes_an_outlier(self):
        x = torch.tensor([[0.0], [1.0], [2.0], [3.0], [100.0]])
        y = torch.tensor([[2.0], [5.0], [8.0], [11.0], [-500.0]])
        weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 0.0])

        weight, bias = centered_ridge(
            x, y, weights=weights, regularization=1.0e-8
        )

        self.assertAlmostEqual(float(weight[0, 0]), 3.0, places=4)
        self.assertAlmostEqual(float(bias[0]), 2.0, places=4)

    def test_weight_shrinkage_enforces_the_effective_sample_floor(self):
        raw = torch.tensor([100.0] + [1.0] * 99)
        shrunk = shrink_weights(raw, parameter_count=25)

        self.assertAlmostEqual(float(shrunk.mean()), 1.0, places=5)
        self.assertGreaterEqual(effective_sample_size(shrunk), 50.0 - 1.0e-4)
        self.assertLess(float(shrunk.max()), float(raw.max() / raw.mean()))

    def test_invalid_weights_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            shrink_weights(torch.tensor([1.0, -1.0]), parameter_count=1)

    def test_vectorized_fit_recovers_each_target_layer_and_head(self):
        generator = np.random.default_rng(9)
        source_values = generator.normal(size=(48, 2, 2, 2)).astype(np.float16)
        target_values = np.empty_like(source_values)
        target_values[:, 0] = source_values[:, 0] * 1.5 + 0.25
        target_values[:, 1] = source_values[:, 1] * -0.75 - 0.5
        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.f16"
            target_path = Path(directory) / "target.f16"
            source_values.tofile(source_path)
            target_values.tofile(target_path)
            source = np.memmap(source_path, mode="r", dtype=np.float16, shape=source_values.shape)
            target = np.memmap(target_path, mode="r", dtype=np.float16, shape=target_values.shape)

            _, _, fp32_r2, bf16_r2 = fit_kind(
                source,
                target,
                torch.tensor([[0], [1]], dtype=torch.int32),
                (0, 36),
                (36, 48),
                1.0e-6,
                "cpu",
                8,
            )

        self.assertGreater(min(fp32_r2), 0.9999)
        self.assertGreater(min(bf16_r2), 0.999)


if __name__ == "__main__":
    unittest.main()
