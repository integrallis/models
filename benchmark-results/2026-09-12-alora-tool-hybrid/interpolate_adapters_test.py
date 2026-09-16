import unittest

from interpolate_adapters import interpolate_state_dicts


class FakeTensor:
    def __init__(self, values, dtype="bf16"):
        self.values = list(values)
        self.shape = (len(values),)
        self.dtype = dtype

    def float(self):
        return FakeTensor(self.values, "float32")

    def lerp(self, other, fraction):
        return FakeTensor(
            [left + (right - left) * fraction for left, right in zip(self.values, other.values)],
            self.dtype,
        )

    def to(self, dtype):
        return FakeTensor(self.values, dtype)


class InterpolateAdaptersTest(unittest.TestCase):
    def test_interpolates_matching_tensors_in_float_and_restores_the_left_dtype(self):
        result = interpolate_state_dicts(
            {"a": FakeTensor([0.0, 2.0])},
            {"a": FakeTensor([10.0, 6.0])},
            0.25,
        )

        self.assertEqual(result["a"].values, [2.5, 3.0])
        self.assertEqual(result["a"].dtype, "bf16")

    def test_rejects_key_shape_or_endpoint_errors(self):
        with self.assertRaisesRegex(ValueError, "strictly between"):
            interpolate_state_dicts({"a": FakeTensor([1])}, {"a": FakeTensor([2])}, 0.0)
        with self.assertRaisesRegex(ValueError, "tensor keys differ"):
            interpolate_state_dicts({"a": FakeTensor([1])}, {"b": FakeTensor([2])}, 0.5)
        with self.assertRaisesRegex(ValueError, "shape differs"):
            interpolate_state_dicts(
                {"a": FakeTensor([1])}, {"a": FakeTensor([2, 3])}, 0.5
            )


if __name__ == "__main__":
    unittest.main()
