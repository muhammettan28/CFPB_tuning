"""Check that relocation preserves recorded identities and resolves real evidence."""
import hashlib
import json
import unittest
from pathlib import Path

from project_paths import ROOT, PAPER, PATH_MAPPING, project_path, recorded_path


class RepositoryLayoutTests(unittest.TestCase):
    def test_historical_paths_resolve_with_component_boundaries(self):
        for old, new in PATH_MAPPING.items():
            self.assertEqual(project_path(old), ROOT/new)
            self.assertEqual(project_path(ROOT/old), ROOT/new)
            self.assertEqual(recorded_path(ROOT/new), old)
        self.assertEqual(project_path('REVISION_PAPER/analysis_extra/file.csv'),
                         PAPER/'analysis_extra/file.csv')
        self.assertEqual(project_path('/tmp/unrelated/file.json'), Path('/tmp/unrelated/file.json'))

    def test_all_frozen_protocol_data_are_available_under_current_layout(self):
        protocol = json.loads((ROOT/'analysis/revision_v1/protocol.json').read_text())
        for old, expected in protocol['files_sha256'].items():
            current = project_path(old)
            self.assertEqual(hashlib.sha256(current.read_bytes()).hexdigest(), expected, old)
            self.assertEqual(recorded_path(current), old)

    def test_pre_relocation_source_snapshots_keep_their_recorded_hashes(self):
        base = ROOT/'experiments/provenance/layout_before_2026_10_26'
        for name, expected in json.loads((base/'SOURCE_HASHES.json').read_text()).items():
            self.assertEqual(hashlib.sha256((base/name).read_bytes()).hexdigest(), expected)
        for model in ['qwen', 'llama', 'bert', 'distilbert', 'roberta']:
            audit = json.loads((ROOT/f'analysis/gpu_preparation/input_audit_{model}.json').read_text())
            for field, name in [('common_code_sha256', 'revision_gpu_common.py'),
                                ('audit_code_sha256', 'audit_gpu_tokenizers.py')]:
                self.assertEqual(hashlib.sha256((base/'scripts'/name).read_bytes()).hexdigest(), audit[field])

    def test_recorded_plans_resolve_to_identical_execution_time_bytes(self):
        frozen = json.loads((ROOT/'analysis/repository_layout/FROZEN_ARTIFACTS.json').read_text())
        plans = [row for row in frozen if row['recorded_path'].startswith('REVISION_PAPER/protocols/')]
        self.assertEqual(len(plans), 2)
        for row in plans:
            archived = project_path(row['recorded_path'])
            self.assertEqual(archived, ROOT/row['current_path'])
            self.assertEqual(hashlib.sha256(archived.read_bytes()).hexdigest(), row['sha256'])
            current = ROOT/'protocols'/archived.name
            self.assertNotEqual(current, archived)
            self.assertTrue(current.is_file())

    def test_paper_directory_contains_only_publication_sources_and_responses(self):
        allowed = {'.tex', '.pdf', '.md', '.cls', '.eps', '.png', '.tif', '.tiff', '.jpg', '.jpeg'}
        responses = {'RESPONSE_TO_REVIEWERS.md', 'REVIEWER_COMMENT_CROSSWALK.md'}
        for path in PAPER.rglob('*'):
            if not path.is_file():
                continue
            self.assertIn(path.suffix, allowed, path)
            if path.suffix == '.md':
                self.assertIn(path.name, responses)
        for name in ['experiments', 'analysis', 'scripts', 'configs', 'literature', 'build']:
            self.assertFalse((PAPER/name).exists())


if __name__ == '__main__':
    unittest.main(verbosity=2)
