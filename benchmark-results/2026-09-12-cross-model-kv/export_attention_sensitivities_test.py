#!/usr/bin/env python3
"""Synthetic tests for receiver-attention sensitivity extraction."""

import unittest

import torch

from export_attention_sensitivities import (
    causal_boundary_attention,
    equal_mass_boundary_sum,
    receiver_sensitivities,
    stacked_host_rows,
)


class AttentionSensitivityTest(unittest.TestCase):
    def test_captured_rows_are_materialized_on_the_host(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        captures = [
            torch.ones((3, 2, 2), device=device),
            torch.full((3, 2, 2), 2.0, device=device),
        ]

        rows = stacked_host_rows(captures)

        self.assertEqual(rows.device.type, "cpu")
        self.assertEqual(tuple(rows.shape), (3, 2, 2, 2))
        self.assertTrue(torch.equal(rows[:, 0], torch.ones((3, 2, 2))))
        self.assertTrue(torch.equal(rows[:, 1], torch.full((3, 2, 2), 2.0)))

    def test_value_and_key_formulas_match_a_direct_calculation(self):
        # Two query heads share one KV head; one future-query boundary, two prefix positions.
        query = torch.tensor([[[2.0, 0.0]], [[0.0, 1.0]]])
        value = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        attention = torch.tensor([[[0.75, 0.25]], [[0.40, 0.60]]])
        output = torch.matmul(attention, value.expand(2, -1, -1))

        key, val = receiver_sensitivities(query, value, attention, output)

        expected_value = torch.square(attention).sum(0).unsqueeze(1)
        expected_key = torch.zeros_like(expected_value)
        for query_head in range(2):
            qnorm = torch.square(query[query_head, 0]).sum() / 2.0
            for position in range(2):
                distance = torch.square(
                    value[0, position] - output[query_head, 0]
                ).sum()
                expected_key[0, 0, position] += (
                    attention[query_head, 0, position] ** 2 * distance * qnorm
                )

        self.assertTrue(torch.allclose(val, expected_value))
        self.assertTrue(torch.allclose(key, expected_key))

    def test_boundary_probe_excludes_the_current_and_future_tokens(self):
        query = torch.zeros(1, 2, 2)
        key = torch.zeros(1, 3, 2)
        value = torch.tensor([[[2.0, 0.0], [0.0, 4.0], [9.0, 9.0]]])

        attention, output = causal_boundary_attention(
            query, key, value, torch.tensor([1, 2]), scaling=1.0
        )

        self.assertTrue(
            torch.allclose(
                attention,
                torch.tensor([[[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]]]),
            )
        )
        self.assertTrue(
            torch.allclose(output, torch.tensor([[[2.0, 0.0], [1.0, 2.0]]]))
        )

    def test_each_boundary_contributes_equal_mass(self):
        key = torch.tensor([[[1.0, 1.0, 0.0]], [[90.0, 10.0, 0.0]]])
        value = torch.tensor([[[3.0, 1.0, 0.0]], [[2.0, 8.0, 0.0]]])

        combined = equal_mass_boundary_sum(
            key, value, torch.tensor([0, 1, 2])
        )

        self.assertTrue(
            torch.allclose(combined[:, 0, 0], torch.tensor([1.4, 0.6, 0.0]))
        )
        self.assertTrue(
            torch.allclose(combined[:, 0, 1], torch.tensor([0.95, 1.05, 0.0]))
        )

    def test_query_heads_must_divide_evenly_across_kv_heads(self):
        with self.assertRaisesRegex(ValueError, "divisible"):
            receiver_sensitivities(
                torch.zeros(3, 1, 2),
                torch.zeros(2, 4, 2),
                torch.zeros(3, 1, 4),
                torch.zeros(3, 1, 2),
            )


if __name__ == "__main__":
    unittest.main()
