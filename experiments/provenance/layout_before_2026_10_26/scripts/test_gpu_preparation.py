"""CPU-only protocol/runner checks; no CUDA probing, weights, training or downloads."""
from __future__ import annotations
import contextlib
import io
import json
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import revision_gpu_common as common
from audit_gpu_tokenizers import separation_failures
from run_revision_gpu import arguments, describe_job, main, selected_encoder


class GPUPreparationTests(unittest.TestCase):
    def test_fixed_membership_matches_frozen_manifests(self):
        common.verify_data()
        for name,count in [('train_10000',10000),('validation_2000',2000)]:
            frame=common.load_frame(common.DATA/f'{name}.csv')
            manifest=pd.read_csv(common.STUDY/f'{name}_manifest.csv')
            self.assertEqual(len(frame),count)
            self.assertEqual(common.selection_ids(frame),common.selection_ids(manifest))
            self.assertTrue(frame.record_id.is_unique)
        plan=common.load_plan()
        for seed in [42,43,44]:
            job=describe_job(arguments(['qlora','--model','qwen','--seed',str(seed)]),plan)
            self.assertFalse(job['resampling'])
            self.assertEqual(job['train_rows'],10000)

    def test_all_catalogue_defaults_avoid_gpu_imports(self):
        before=set(sys.modules)
        count=0
        for line in (common.PREP/'commands_dry_run.txt').read_text().splitlines():
            argv=shlex.split(line)[2:]
            if argv[0]=='audit':
                continue  # Actual tokenizer audit is CPU-only but requires resolved snapshots.
            capture=io.StringIO()
            with contextlib.redirect_stdout(capture): main(argv)
            result=json.loads(capture.getvalue())
            self.assertFalse(result['gpu_started'])
            self.assertFalse(result['execution_requested'])
            count+=1
        self.assertGreater(count,60)
        self.assertNotIn('torch',set(sys.modules)-before)
        self.assertNotIn('trl',set(sys.modules)-before)
        self.assertNotIn('peft',set(sys.modules)-before)

    def test_unresolved_registry_blocks_artifact_use(self):
        for model in common.load_plan()['models']:
            with self.assertRaisesRegex(RuntimeError,'immutable'):
                common.artifact_entry(common.PAPER/'configs/model_registry.template.json',model)

    def test_tampered_data_fails_before_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            data=root/'REVISION_PAPER/data/revision_v1';data.mkdir(parents=True)
            (data/'train_10000.csv').write_text('tampered')
            with patch.object(common,'ROOT',root),patch.object(common,'DATA',data):
                with self.assertRaisesRegex(RuntimeError,'Frozen dataset changed'):
                    common.verify_data()

    def test_global_tie_rule_is_not_pairwise_chaining(self):
        rows=[{'candidate_id':1,'validation_macro_f1':.8},
              {'candidate_id':2,'validation_macro_f1':.8+0.8e-12},
              {'candidate_id':3,'validation_macro_f1':.8+1.6e-12}]
        self.assertEqual(common.pick_candidate(rows)['candidate_id'],2)

    def test_selection_requires_complete_search_and_frozen_winner(self):
        import run_revision_gpu as runner
        plan=common.load_plan()
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner,'EXP',Path(tmp)):
            out=Path(tmp)/'encoder/bert/search';out.mkdir(parents=True)
            rows=[{**c,'validation_macro_f1':.8} for c in plan['encoder']['grid']]
            chosen={'candidate':plan['encoder']['grid'][0], 'plan_sha256':common.sha_file(common.CONFIG)}
            config=out/'selected_config.json'
            (out/'validation_search.json').write_text(common.json_text(rows[:-1]))
            chosen['validation_search_sha256']=common.sha_file(out/'validation_search.json')
            config.write_text(common.json_text(chosen))
            (out/'selected_config.sha256').write_text(common.sha_file(config)+'  selected_config.json\n')
            (out/'validation_search.json').write_text(common.json_text(rows[:-1]))
            with self.assertRaisesRegex(RuntimeError,'incomplete'): selected_encoder('bert',plan)
            (out/'validation_search.json').write_text(common.json_text(rows))
            chosen['validation_search_sha256']=common.sha_file(out/'validation_search.json')
            config.write_text(common.json_text(chosen))
            (out/'selected_config.sha256').write_text(common.sha_file(config)+'  selected_config.json\n')
            self.assertEqual(selected_encoder('bert',plan)[0]['candidate']['candidate_id'],1)
            config.write_text(common.json_text({**chosen,'candidate':plan['encoder']['grid'][1]}))
            with self.assertRaisesRegex(RuntimeError,'hash mismatch'): selected_encoder('bert',plan)

    def test_token_overlap_is_label_blind(self):
        keys=pd.DataFrame([{'role':r,'token_hash':k,'label':l} for r,k,l in [
            ('train','a','credit_card'),('validation','b','credit_card'),
            ('internal_test','a','retail_banking'),('legacy','c','credit_reporting')]])
        failures=separation_failures(keys)
        self.assertEqual(failures,[{'left':'train','right':'internal_test','shared':1}])

    def test_latency_cohort_balance_and_frozen_identifiers(self):
        manifest=pd.read_csv(common.PREP/'latency_sample_manifest.csv')
        test=common.load_frame(common.DATA/'internal_test.csv')
        self.assertEqual(len(manifest),100)
        self.assertEqual(set(manifest['product'].value_counts()),{20})
        self.assertTrue(manifest.record_id.is_unique)
        for label in common.LABELS:
            rows=test.loc[test['product'].eq(label)].copy()
            rows['key']=rows.record_id.map(lambda r:common.sha_text(f'latency_v1|{r}'))
            expected=set(rows.sort_values('key').head(20).record_id)
            self.assertEqual(expected,set(manifest.loc[manifest['product'].eq(label),'record_id']))
        prep=json.loads((common.PREP/'preparation_manifest.json').read_text())
        self.assertEqual(common.sha_file(common.PREP/'latency_sample_manifest.csv'),prep['latency_sample_sha256'])

    def test_cached_tokenizer_sft_masks_and_no_duplicate_special_tokens(self):
        from transformers import AutoTokenizer
        examples=common.load_frame(common.DATA/'train_10000.csv').head(10)
        paths=['experiment_archive/outputs/qwen3_8b_qlora_pilot_5952287/best_adapter',
               'experiment_archive/outputs/llama31_8b_fixed_protocol_seed42_5994012_0/best_adapter']
        for path in paths:
            tok=AutoTokenizer.from_pretrained(common.ROOT/path,local_files_only=True)
            for row in examples.itertuples(index=False):
                rendered,encoded=common.llm_encoding(tok,row.narrative,500,512)
                template_ids=tok.apply_chat_template(common.messages(row.narrative,500),tokenize=True,
                                                      add_generation_prompt=True,enable_thinking=False)
                self.assertEqual(encoded['input_ids'],template_ids)
                sft=common.sft_record(tok,row.narrative,row.product)
                boundary=len(encoded['input_ids'])
                self.assertEqual(sft['input_ids'][:boundary],encoded['input_ids'])
                self.assertEqual(sft['completion_mask'][:boundary],[0]*boundary)
                self.assertGreater(sum(sft['completion_mask']),0)
                self.assertLessEqual(len(sft['input_ids']),512)
        # Cached pilot/tokenizer examples test serialization only, not final artifact identity.

    def test_invalid_model_action_and_condition_fail(self):
        plan=common.load_plan()
        for argv in [['qlora','--model','bert'],['encoder-search','--model','qwen'],
                     ['evaluate','--model','bert','--variant','base'],
                     ['long-input','--model','qwen','--condition','unregistered']]:
            with self.assertRaises(ValueError):describe_job(arguments(argv),plan)

    def test_simulated_multiple_gpu_execution_stops_before_output(self):
        # This is a fake CUDA object, not a CUDA probe or a real --execute-gpu job.
        import run_revision_gpu as runner
        fake=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:True,device_count=lambda:2))
        with (patch.object(runner,'audit_guard',return_value=({'repo_id':'Qwen/Qwen3-8B'},Path('/unused'),'audit')),
              patch.object(runner,'versions',return_value={'transformers':'4.57.6','trl':'0.23.1','peft':'0.17.1'}),
              patch.dict(sys.modules,{'torch':fake})):
            with self.assertRaisesRegex(RuntimeError,'exactly one'):
                runner.main(['qlora','--model','qwen','--execute-gpu'])


if __name__=='__main__':unittest.main(verbosity=2)
