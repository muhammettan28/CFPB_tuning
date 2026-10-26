import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import auto_postprocess_after_experiments as auto


class AutomaticContinuationTests(unittest.TestCase):
    def setup_tree(self,root):
        paper=root/'REVISION_PAPER';core=paper/'analysis/gpu_execution';extra=paper/'analysis/ref14_encoder';out=paper/'analysis/postprocessing'
        for p in [core,extra,out,paper/'configs']:p.mkdir(parents=True,exist_ok=True)
        main=paper/'configs/gpu_revision_v1.json';main.write_text('{}')
        reference=paper/'configs/ref14_encoder_v1.json';reference.write_text('{}')
        (core/'GPU_RUN_STATUS.json').write_text(json.dumps({'status':'experiments_and_analysis_completed'}))
        (extra/'RUN_STATUS.json').write_text(json.dumps({'status':'waiting_for_core_study'}))
        (core/'GPU_RESULT_VALIDATION.json').write_text(json.dumps({'gpu_experiments_completed':True,'all_88_gpu_jobs_complete':True,
            'protocol_sha256':auto.PROTOCOL_SHA,'plan_sha256':auto.sha_file(main)}))
        (extra/'RESULT_VALIDATION.json').write_text(json.dumps({'completed':True,'all_12_gpu_jobs_complete':True,'configuration_sha256':auto.sha_file(reference)}))
        return paper,core,extra,out,main

    def test_core_completion_does_not_start_work_before_reference_finishes(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper,core,extra,out,main=self.setup_tree(Path(tmp))
            with patch.multiple(auto,PAPER=paper,CORE=core,EXTRA=extra,OUT=out,CONFIG=main):
                self.assertFalse(auto.readiness()[0])
                (extra/'RUN_STATUS.json').write_text(json.dumps({'status':'experiments_and_analysis_completed'}))
                self.assertTrue(auto.readiness()[0])
                (out/'CANCEL_AUTO_START').touch()
                self.assertEqual(auto.readiness(),(False,'cancelled_by_marker'))

    def test_changed_protocol_hash_blocks_a_completed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            paper,core,extra,out,main=self.setup_tree(Path(tmp))
            (extra/'RUN_STATUS.json').write_text(json.dumps({'status':'experiments_and_analysis_completed'}))
            main.write_text('{"changed": true}')
            with patch.multiple(auto,PAPER=paper,CORE=core,EXTRA=extra,OUT=out,CONFIG=main):
                self.assertEqual(auto.readiness(),(False,'core_validation_inconsistent'))

    def test_receipt_requires_real_deliverables_with_unchanged_hashes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);paper,core,extra,out,main=self.setup_tree(root)
            manifest=paper/'analysis/revision_v1';manifest.mkdir()
            (manifest/'source_integrity_before.json').write_text('{}')
            info={k:True for k in ['draft_package_ready_for_author_review','manuscript_integrated','supplementary_created',
                'response_letter_created','pdf_visual_qa_passed','originals_unchanged','package_hashes_verified']}
            info.update(uploaded=False,author_final_review_pending=True,artifacts_sha256={})
            receipt=out/'POSTPROCESS_VALIDATION.json';receipt.write_text(json.dumps(info))
            with patch.multiple(auto,ROOT=root,PAPER=paper,DONE=receipt):
                self.assertFalse(auto.finished()[0])
                for name in ['manuscript.tex','manuscript.pdf','supplementary_material.tex','supplementary_material.pdf',
                    'RESPONSE_TO_REVIEWERS.md','response_to_reviewers.tex','response_to_reviewers.pdf','reviewer.zip']:
                    p=paper/name;p.write_text('test-artifact')
                    info['artifacts_sha256'][str(p.relative_to(root))]=auto.sha_file(p)
                receipt.write_text(json.dumps(info));self.assertTrue(auto.finished()[0])
                (paper/'manuscript.pdf').write_text('changed after verification')
                self.assertFalse(auto.finished()[0])

    def test_resume_targets_exact_thread_with_workspace_permissions(self):
        settings={'codex_cli':'/bin/true','thread_id':'explicit-session-id'}
        command=auto.command(settings,1)
        self.assertIn('resume',command);self.assertIn('explicit-session-id',command)
        self.assertNotIn('--last',command);self.assertNotIn('--dangerously-bypass-approvals-and-sandbox',command)
        self.assertIn('workspace-write',command);self.assertIn('never',command)

    def test_fork_avoids_parent_writer_and_later_resumes_only_the_followup(self):
        settings={'codex_cli':'/bin/true','thread_id':'parent-id','continuation_mode':'fork'}
        first=auto.command(settings,1)
        self.assertIn('fork',first);self.assertIn('parent-id',first)
        settings['continuation_thread_id']='followup-id'
        later=auto.command(settings,2)
        self.assertIn('resume',later);self.assertIn('followup-id',later);self.assertNotIn('parent-id',later)

    def test_session_id_is_taken_only_from_the_structured_start_event(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'events.jsonl'
            path.write_text('warning: not JSON\n'+json.dumps({'type':'thread.started','thread_id':'followup-id'})+'\n')
            self.assertEqual(auto.started_thread(path),'followup-id')


if __name__=='__main__':unittest.main(verbosity=2)
