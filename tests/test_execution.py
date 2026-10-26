import json
from pathlib import Path
import tempfile
import unittest
import zipfile

import numpy as np

from repro.cli import STUDY, inspect_submission, legacy_run, read_panel, reference_baselines


def fixture(root):
    """Synthetic inputs exercise the actual recovered kNN and logistic runners."""
    rng = np.random.default_rng(1234)
    n = 360
    fam = np.repeat(np.arange(3), n // 3)
    features = np.eye(3)[fam] + rng.normal(0, .05, (n, 3))
    rewards = np.full((n, 3), .1)
    rewards[np.arange(n), fam] = .9
    costs = np.broadcast_to(np.array([.0002, .0005, .001]), (n, 3)).copy()
    train = np.concatenate([np.arange(g * 120, g * 120 + 80) for g in range(3)])
    test = np.setdiff1d(np.arange(n), train)
    fit = np.concatenate([np.arange(g * 120, g * 120 + 60) for g in range(3)])
    val = np.setdiff1d(train, fit)
    directory = root / STUDY / 'inputs/routerbench-0shot'
    directory.mkdir(parents=True)
    np.savez_compressed(directory / 'panel.npz', r=rewards, c=costs, fam=fam,
                        query_ids=np.array([f'synthetic-{i}' for i in range(n)]))
    (directory / 'source_manifest.json').write_text('{"synthetic": true}')
    split = {'train': train.tolist(), 'test': test.tolist(),
             'inner': [{'fit': fit.tolist(), 'val': val.tolist()}]}
    (directory / 'splits.json').write_text(json.dumps({'iid_0': split}))
    fdir = root / 'embeddings/routerbench-0shot'
    fdir.mkdir(parents=True)
    np.save(fdir / 'gte.npy', features.astype(np.float32))
    return directory


class ExecutionTests(unittest.TestCase):
    def test_rejects_held_out_rows_in_inner_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            directory = fixture(root)
            splits = json.loads((directory / 'splits.json').read_text())
            splits['iid_0']['inner'][0]['val'][0] = splits['iid_0']['test'][0]
            (directory / 'splits.json').write_text(json.dumps(splits))
            with self.assertRaisesRegex(ValueError, 'outside training'):
                read_panel(root, 'routerbench-0shot', 'any')

    def test_reference_baselines_have_expected_utilities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            out = root / 'reference.json'
            reference_baselines(root, 'routerbench-0shot', 'iid_0', out, 'any')
            curves = json.loads(out.read_text())['results']['iid_0']
            self.assertAlmostEqual(curves['0']['best_single_train']['utility'], (0.9 + 0.1 + 0.1) / 3)
            self.assertAlmostEqual(curves['0']['realized_oracle']['utility'], .9)
            self.assertAlmostEqual(curves['100']['realized_oracle']['utility'], .9 - 100 * (.0002 + .0005 + .001) / 3)
            with self.assertRaises(FileExistsError):
                reference_baselines(root, 'routerbench-0shot', 'iid_0', out, 'any')

    def test_original_knn_and_prior_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture(root)
            for kind, method in [('knn', 'knn100'), ('prior', 'soft_prior')]:
                launched = legacy_run(root, 'routerbench-0shot', 'iid_0', 'gte', kind, root / kind, 'any')
                result = json.loads(Path(launched['result']).read_text())
                self.assertAlmostEqual(result['results']['0'][method]['utility'], .9)
                self.assertEqual(len(result['results']), 13)
                with np.load(Path(launched['result']).with_name('predictions.npz'), allow_pickle=False) as z:
                    self.assertEqual(len(z['test_rows']), 120)

    def test_archive_inspection_does_not_extract_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive = root / 'submission.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../escape.py', 'raise RuntimeError("never execute")')
                z.writestr('package/requirements.txt', 'numpy')
            result = inspect_submission(archive)
            self.assertEqual(result['unsafe_member_paths'], ['../escape.py'])
            self.assertEqual(result['python_sources'], 1)
            self.assertFalse((root / 'escape.py').exists())


if __name__ == '__main__':
    unittest.main()
