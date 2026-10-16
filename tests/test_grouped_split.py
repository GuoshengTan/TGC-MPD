import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from train_treecross import split_by_package_name


class PackageNameSplitTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {'package_name': f'@scope/pkg-{group}', 'version': str(version), 'label': group % 2, 'row_id': group * 3 + version}
            for group in range(120) for version in range(1 + group % 3)
        ])

    def test_groups_are_disjoint_and_all_rows_are_retained(self):
        splits = split_by_package_name(self.frame, 40, .2, .1)
        names = [set(part.package_name) for part in splits]
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertFalse(names[i] & names[j])
        self.assertEqual(set(pd.concat(splits).row_id), set(self.frame.row_id))
        self.assertEqual(sum(len(part) for part in splits), len(self.frame))
        self.assertTrue(all(part.label.nunique() == 2 for part in splits))

    def test_same_seed_is_reproducible(self):
        left = split_by_package_name(self.frame, 42, .2, .1)
        right = split_by_package_name(self.frame, 42, .2, .1)
        for a, b in zip(left, right):
            pd.testing.assert_frame_equal(a, b)

    def test_missing_names_and_unsupported_proportions_are_rejected(self):
        missing = self.frame.copy()
        missing.loc[0, 'package_name'] = None
        with self.assertRaises(ValueError):
            split_by_package_name(missing, 40, .2, .1)
        with self.assertRaises(ValueError):
            split_by_package_name(self.frame, 40, .3, .1)


if __name__ == '__main__':
    unittest.main()

