import unittest

from spn_accel_cmodel.validation import MetricTolerance, compare_metrics, flatten_metrics


class ValidationTest(unittest.TestCase):
    def test_metric_thresholds(self):
        report = compare_metrics(
            "x",
            {"cycles": 104, "bytes": 100},
            {"cycles": 100, "bytes": 100},
            tolerances={"cycles": MetricTolerance(relative=0.05)},
        )
        self.assertTrue(report.passed)
        report2 = compare_metrics(
            "x",
            {"cycles": 106},
            {"cycles": 100},
            tolerances={"cycles": MetricTolerance(relative=0.05)},
        )
        self.assertFalse(report2.passed)

    def test_flatten(self):
        self.assertEqual(flatten_metrics({"a": {"b": 3}, "c": 2.0}), {"a.b": 3.0, "c": 2.0})


if __name__ == "__main__":
    unittest.main()
