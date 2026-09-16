#!/usr/bin/env python3
"""Unit tests for exact-answer retention gates."""

import unittest

from evaluate_hf_retrieval import exact_answer_retention


class ExactAnswerRetentionTest(unittest.TestCase):
    def test_requires_every_native_correct_answer_to_remain_correct(self):
        records = [
            {"gold": 1, "nativeWinner": 1, "mappedWinner": 1},
            {"gold": 0, "nativeWinner": 0, "mappedWinner": 2},
            {"gold": 3, "nativeWinner": 1, "mappedWinner": 3},
        ]

        actual = exact_answer_retention(records)

        self.assertEqual(actual["nativeCorrect"], 2)
        self.assertEqual(actual["retainedNativeCorrect"], 1)
        self.assertEqual(actual["retention"], 0.5)
        self.assertFalse(actual["pass"])

    def test_rejects_a_vacuous_gate_when_native_has_no_correct_answer(self):
        actual = exact_answer_retention(
            [{"gold": 1, "nativeWinner": 0, "mappedWinner": 1}]
        )

        self.assertEqual(actual["nativeCorrect"], 0)
        self.assertFalse(actual["pass"])


if __name__ == "__main__":
    unittest.main()
