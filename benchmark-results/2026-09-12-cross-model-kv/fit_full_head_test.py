#!/usr/bin/env python3
"""Synthetic tests for the full cross-head KV mapper fitter."""

import hashlib
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from fit_full_head import fit, fit_full_kind


class FullHeadFitterTest(unittest.TestCase):
    def test_fit_recovers_cross_head_dependencies(self):
        generator = np.random.default_rng(41)
        source_values = generator.normal(size=(96, 2, 2, 2)).astype(np.float16)
        target_values = np.empty_like(source_values)
        # Every target head depends on the other source head. A head-local fit cannot recover this.
        target_values[:, 0, 0, 0] = 2.0 * source_values[:, 0, 1, 0] + 0.25
        target_values[:, 0, 0, 1] = -3.0 * source_values[:, 0, 1, 1] - 0.5
        target_values[:, 0, 1, 0] = 0.75 * source_values[:, 0, 0, 0] + 1.0
        target_values[:, 0, 1, 1] = 1.5 * source_values[:, 0, 0, 1] - 1.25
        target_values[:, 1] = source_values[:, 1] * -0.5 + 0.75

        with TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.f16"
            target_path = Path(directory) / "target.f16"
            source_values.tofile(source_path)
            target_values.tofile(target_path)
            source = np.memmap(source_path, mode="r", dtype=np.float16, shape=source_values.shape)
            target = np.memmap(target_path, mode="r", dtype=np.float16, shape=target_values.shape)

            _, _, fp32_r2, bf16_r2 = fit_full_kind(
                source,
                target,
                torch.tensor([[0], [1]], dtype=torch.int32),
                (0, 72),
                (72, 96),
                1.0e-6,
                "cpu",
                16,
            )

        self.assertGreater(min(fp32_r2), 0.9999)
        self.assertGreater(min(bf16_r2), 0.999)

    def test_fit_accepts_exported_attention_identity_and_records_weighting(self):
        generator = np.random.default_rng(91)
        source_values = generator.normal(size=(12, 1, 1, 1)).astype(np.float16)
        target_values = (source_values * 1.25 + 0.5).astype(np.float16)
        sensitivities = np.ones((8, 1, 1, 2), dtype=np.float32)

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            target_root = root / "target"
            attention_root = root / "attention"
            out = root / "out"
            for path in (source_root, target_root, attention_root):
                path.mkdir()
            sequences = root / "sequences.jsonl"
            sequences.write_text(
                json.dumps({"seq_id": "calibration-0", "split": "calibration"})
                + "\n"
                + json.dumps({"seq_id": "validation-0", "split": "validation"})
                + "\n"
            )
            sequence_sha = hashlib.sha256(sequences.read_bytes()).hexdigest()
            for model_root, model_id, revision, values in (
                (source_root, "source/model", "source-revision", source_values),
                (target_root, "target/model", "target-revision", target_values),
            ):
                files = {}
                for kind in ("keys", "values"):
                    path = model_root / f"{kind}.f16"
                    values.tofile(path)
                    files[kind] = {"name": path.name}
                (model_root / "manifest.json").write_text(
                    json.dumps(
                        {
                            "shape": list(values.shape),
                            "source": {"model_id": model_id, "revision": revision},
                            "sampledPositions": list(range(8)),
                            "splits": {"calibration": [0, 8], "validation": [8, 12]},
                            "files": files,
                        }
                    )
                )
            sensitivity_path = attention_root / "sensitivities.f32"
            sensitivities.tofile(sensitivity_path)
            (attention_root / "manifest.json").write_text(
                json.dumps(
                    {
                        "layout": "observation,target-layer,kv-head,key-or-value",
                        "dtype": "float32",
                        "model": "target/model",
                        "revision": "target-revision",
                        "sequenceManifestSha256": sequence_sha,
                        "sequenceIds": ["calibration-0"],
                        "sampledPositions": list(range(8)),
                        "shape": list(sensitivities.shape),
                        "file": {
                            "name": sensitivity_path.name,
                            "bytes": sensitivity_path.stat().st_size,
                            "sha256": hashlib.sha256(sensitivity_path.read_bytes()).hexdigest(),
                        },
                    }
                )
            )
            selection = root / "selection.json"
            selection.write_text(json.dumps({"selection": [[0]]}))

            report = fit(
                source_root,
                target_root,
                selection,
                sequences,
                out,
                1.0e-6,
                "cpu",
                4,
                attention_root,
            )

        self.assertIsNotNone(report["attentionWeighting"])
        self.assertEqual(
            report["attentionWeighting"]["parameterCountPerTargetHead"], 1
        )

    def test_per_head_weights_exclude_only_that_heads_outliers(self):
        generator = np.random.default_rng(73)
        source_values = generator.normal(size=(96, 1, 2, 1)).astype(np.float16)
        target_values = np.empty_like(source_values)
        target_values[:, 0, 0, 0] = 2.0 * source_values[:, 0, 1, 0] + 0.25
        target_values[:, 0, 1, 0] = -1.5 * source_values[:, 0, 0, 0] - 0.4
        target_values[:8, 0, 0, 0] += 100.0
        observation_weights = np.ones((72, 1, 2), dtype=np.float32)
        observation_weights[:8, 0, 0] = 0.0

        with TemporaryDirectory() as directory:
            root = Path(directory)
            source_path = root / "source.f16"
            target_path = root / "target.f16"
            weight_path = root / "weights.f32"
            source_values.tofile(source_path)
            target_values.tofile(target_path)
            observation_weights.tofile(weight_path)
            source = np.memmap(
                source_path, mode="r", dtype=np.float16, shape=source_values.shape
            )
            target = np.memmap(
                target_path, mode="r", dtype=np.float16, shape=target_values.shape
            )
            weights = np.memmap(
                weight_path, mode="r", dtype=np.float32, shape=observation_weights.shape
            )

            _, _, fp32_r2, bf16_r2 = fit_full_kind(
                source,
                target,
                torch.tensor([[0]], dtype=torch.int32),
                (0, 72),
                (72, 96),
                1.0e-6,
                "cpu",
                16,
                observation_weights=weights,
            )

        self.assertGreater(min(fp32_r2), 0.9999)
        self.assertGreater(min(bf16_r2), 0.999)


if __name__ == "__main__":
    unittest.main()
