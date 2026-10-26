"""Scientific checks for off-label metrics, dependency groups and resampling."""
import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from audit_revision_data import dependency_groups
from prepare_revision_protocol import assign_roles
from analyze_revision_truncation import match_views
from analyze_archived_predictions import (
    LABELS, confusion_metrics, encode, group_confusions, paired_cluster_bootstrap,
)


class RevisionStatisticsTests(unittest.TestCase):
    def test_ambiguous_prefix_views_are_not_arbitrarily_paired(self):
        common={"seed":1,"condition":"balanced","true_label":"credit_card"}
        records=[
            {**common,"max_chars":250,"source_file":"short.csv","source_row":0,"narrative":"a"*250},
            {**common,"max_chars":250,"source_file":"short.csv","source_row":1,"narrative":"c"*250},
            {**common,"max_chars":500,"source_file":"long.csv","source_row":0,"narrative":"a"*250+" b"},
            {**common,"max_chars":500,"source_file":"long.csv","source_row":1,"narrative":"a"*250+" d"},
            {**common,"max_chars":500,"source_file":"long.csv","source_row":2,"narrative":"c"*250+" e"},
        ]
        pairs,counts=match_views(pd.DataFrame(records))
        self.assertEqual(len(pairs),1)
        self.assertEqual(int(pairs.iloc[0].short_row),1)
        self.assertEqual(int(counts.iloc[0].ambiguous_long_rows),2)

    def test_evaluation_priority_and_validation_priority(self):
        frame = pd.DataFrame({"original_role": ["train", "validation", "internal_test", "train", "validation", "train", "legacy"],
                              "group": [0, 0, 0, 1, 1, 2, 3]})
        role, reason = assign_roles(frame)
        self.assertEqual(role.tolist(), ["quarantine", "quarantine", "internal_test", "quarantine", "validation", "train", "legacy"])
        self.assertEqual(reason[3], "component_touches_retained_validation")

    def test_off_label_errors_and_sample_weights_match_sklearn(self):
        true = np.array([LABELS[0], LABELS[0], LABELS[1], LABELS[2], LABELS[3], LABELS[4]])
        pred = np.array(["__off_label__", LABELS[0], LABELS[0], LABELS[2], LABELS[3], LABELS[4]])
        weights = np.array([.5, .5, 1., 1., 1., 1.])
        cm = group_confusions(encode(true), encode(pred[:, None]), np.arange(len(true)), weights).sum(axis=0)
        actual = confusion_metrics(cm)[0]
        expected = [accuracy_score(true, pred, sample_weight=weights),
                    f1_score(true, pred, labels=LABELS, average="macro", sample_weight=weights, zero_division=0),
                    f1_score(true, pred, labels=LABELS, average="weighted", sample_weight=weights, zero_division=0)]
        np.testing.assert_allclose(actual, expected, atol=1e-14)

    def test_cluster_bootstrap_preserves_sizes_and_pairing(self):
        # Two clusters: two correct rows in cluster A, one incorrect row in B.
        # Resampling TWO clusters gives BB (accuracy 0), AB (2/3), AA (1).
        # Their probabilities are exactly 1/4, 1/2, 1/4.
        y = np.array([0, 0, 1])
        pred = np.array([[0, 0], [0, 0], [0, 0]])
        cm = group_confusions(y, pred, np.array([0, 0, 1]))
        point, samples = paired_cluster_bootstrap(cm, 20000, 42)
        self.assertAlmostEqual(point[0, 0], 2 / 3)
        np.testing.assert_array_equal(samples[:, 0], samples[:, 1])
        values, frequencies = np.unique(samples[:, 0, 0], return_counts=True)
        np.testing.assert_allclose(values, [0, 2 / 3, 1])
        np.testing.assert_allclose(frequencies / len(samples), [.25, .5, .25], atol=.015)

    def test_transitive_dependencies_ignore_labels(self):
        frame = pd.DataFrame({"full": ["a", "a", "b", "c"],
                              "prefix": ["x", "y", "y", "z"],
                              "label": [0, 1, 2, 0]})
        groups = dependency_groups(frame, ["full", "prefix"])
        self.assertEqual(groups[0], groups[2])
        self.assertNotEqual(groups[0], groups[3])


if __name__ == "__main__":
    unittest.main()
