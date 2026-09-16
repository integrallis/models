#!/usr/bin/env python3
"""Unit tests for cross-model KV task-quality scoring and gates."""

import math
import unittest

from evaluate_hf_quality import (
    choice_metrics,
    handoff_partition,
    log_probability,
    perplexity_summary,
    sequence_log_probability,
)


class QualityScoringTest(unittest.TestCase):
    def test_handoff_partition_maps_the_old_prefix_and_recomputes_the_recent_tail(self):
        sink, mapped, target = handoff_partition([10, 11, 12, 13, 14, 15], 1, 2)

        self.assertEqual(sink, [10])
        self.assertEqual(mapped, [11, 12, 13])
        self.assertEqual(target, [14, 15])

    def test_handoff_partition_requires_a_nonempty_mapped_prefix(self):
        with self.assertRaisesRegex(ValueError, "smaller than the context"):
            handoff_partition([10, 11], 0, 2)

    def test_log_probability_uses_log_softmax_for_the_requested_token(self):
        logits = [0.0, 1.0, 2.0]

        actual = log_probability(logits, 2)

        expected = math.log(math.exp(2.0) / sum(math.exp(x) for x in (0.0, 1.0, 2.0)))
        self.assertAlmostEqual(actual, expected, places=6)

    def test_choice_metrics_reports_raw_and_length_normalized_winners(self):
        scores = [(-4.0, 1), (-5.0, 5), (-6.0, 2), (-7.0, 3)]

        actual = choice_metrics(scores, label=1)

        self.assertEqual(actual["rawWinner"], 0)
        self.assertEqual(actual["normalizedWinner"], 1)
        self.assertFalse(actual["rawCorrect"])
        self.assertTrue(actual["normalizedCorrect"])

    def test_sequence_log_probability_pairs_each_position_with_its_token(self):
        logits = [[2.0, 0.0], [0.0, 3.0]]

        actual = sequence_log_probability(logits, [0, 1])

        expected = log_probability(logits[0], 0) + log_probability(logits[1], 1)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_sequence_log_probability_rejects_misaligned_rows(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            sequence_log_probability([[1.0, 2.0]], [0, 1])

    def test_perplexity_gate_compares_paired_total_nll_not_mean_per_chunk(self):
        actual = perplexity_summary(
            native_nll=[10.0, 1.0],
            mapped_nll=[11.0, 1.1],
            token_counts=[10, 1],
            maximum_ratio=1.25,
        )

        self.assertAlmostEqual(actual["native"]["meanNll"], 1.0)
        self.assertAlmostEqual(actual["mapped"]["meanNll"], 1.1)
        self.assertAlmostEqual(actual["pplRatio"], math.exp(0.1), places=6)
        self.assertTrue(actual["gate"]["pass"])

    def test_perplexity_gate_rejects_misaligned_inputs(self):
        with self.assertRaisesRegex(ValueError, "aligned"):
            perplexity_summary([1.0], [1.0, 2.0], [1], 1.25)


if __name__ == "__main__":
    unittest.main()
