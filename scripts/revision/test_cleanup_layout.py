import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import revision_document_paths as documents
import auto_postprocess_after_experiments as auto
import run_gpu_study as core
import run_ref14_baselines as reference


class CleanupLayoutTests(unittest.TestCase):
    def test_compiled_evidence_prefers_build_and_supports_older_archives(self):
        with tempfile.TemporaryDirectory() as temp:
            paper=Path(temp)
            old=paper/'manuscript.aux';old.write_text('old')
            with patch.multiple(documents,PAPER=paper,LABELS=paper/'labels'):
                self.assertEqual(documents.compiled_artifact('manuscript','.aux'),old)
                new=paper/'build/manuscript.aux';new.parent.mkdir();new.write_text('new')
                self.assertEqual(documents.compiled_artifact('manuscript','.aux'),new)
                with self.assertRaises(ValueError):documents.compiled_artifact('../other','.aux')

    def test_finished_workflow_cannot_resume_a_retired_drafting_task(self):
        with tempfile.TemporaryDirectory() as temp,patch.object(auto,'PAPER',Path(temp)),patch.object(auto.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'retired'):auto.continue_revision({})
            launch.assert_not_called()

    def test_monitor_writers_do_not_recreate_top_level_status_documents(self):
        with tempfile.TemporaryDirectory() as temp:
            paper=Path(temp);out=paper/'analysis/runtime';out.mkdir(parents=True)
            state={'status':'experiments_and_analysis_completed','completed_jobs':88,
                   'total_gpu_jobs':88,'plan_sha256':'test'}
            with patch.multiple(core,PAPER=paper,EXEC=out):core.status(state)
            with patch.multiple(reference,PAPER=paper,OUT=out):reference.state('experiments_and_analysis_completed')
            with patch.multiple(auto,PAPER=paper,OUT=out):auto.write_status('draft_package_ready_for_author_review')
            for name in ['GPU_RUN_STATUS.md','REF14_BASELINE_STATUS.md','AUTO_POSTPROCESS_STATUS.md']:
                self.assertTrue((out/name).is_file())
                self.assertFalse((paper/name).exists())


if __name__=='__main__':unittest.main(verbosity=2)
