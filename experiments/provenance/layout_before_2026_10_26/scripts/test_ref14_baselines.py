import tempfile
import unittest
import json
import sys
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import run_ref14_baselines as ref
import run_revision_gpu as runner
import analyze_ref14_baselines as analysis
import numpy as np


class ReferenceBaselineExecutionTests(unittest.TestCase):
    def test_paired_analysis_does_not_mix_model_or_seed_predictions(self):
        labels=runner.LABELS+runner.LABELS[:3]
        target=pd.DataFrame({'true_label':labels,'group':list(range(8))})
        def added(model,seed,target,role):
            result=np.asarray(labels).copy()
            if model=='bert' and seed==42:result[0]=runner.LABELS[1]
            return result
        with patch.object(analysis,'config',return_value={'models':{'bert':{},'roberta':{}}}),patch.object(analysis.core,'source_predictions',return_value=np.asarray(labels)),patch.object(analysis,'predictions',side_effect=added):
            rows,contrasts=analysis.summarise(target,'internal')
        macro=[r for r in contrasts if r['metric']=='macro_f1']
        self.assertEqual(next(r['ref14_minus_main'] for r in macro if r['model']=='roberta'),0)
        self.assertLess(next(r['ref14_minus_main'] for r in macro if r['model']=='bert'),0)
        self.assertEqual(len(rows),12)

    def test_dataset_cap_override_preserves_default_and_character_budget(self):
        frame=pd.DataFrame({'narrative':['a'*600],'product':[runner.LABELS[0]]})
        seen=[]
        def tokenize(texts,**kwargs):
            seen.append((texts,kwargs));return {'input_ids':[[1,2]]}
        module=SimpleNamespace(Dataset=SimpleNamespace(from_dict=lambda d:d))
        with patch.dict(sys.modules,{'datasets':module}):
            runner.encoder_dataset(frame,tokenize)
            runner.encoder_dataset(frame,tokenize,max_length=100)
        self.assertEqual([x[1]['max_length'] for x in seen],[512,100])
        self.assertTrue(all(len(x[0][0])==500 for x in seen))

    def test_active_or_failed_core_never_starts_additional_gpu_job(self):
        for core_status in ['running','failed']:
            with self.subTest(status=core_status),tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);core=root/'analysis/gpu_execution';core.mkdir(parents=True)
                (core/'GPU_RUN_STATUS.json').write_text(json.dumps({'status':core_status}))
                out=root/'reference'
                with patch.object(ref,'PAPER',root),patch.object(ref,'OUT',out),patch.object(ref,'config',return_value={'models':{'bert':{},'roberta':{}}}),patch.object(ref,'guard'),patch.object(ref.subprocess,'Popen') as launch:
                    if core_status=='running':
                        with self.assertRaisesRegex(RuntimeError,'still active'):ref.run_study(False)
                    else:
                        ref.run_study(False)
                        self.assertEqual(json.loads((out/'RUN_STATUS.json').read_text())['status'],'blocked_core_not_complete')
                    launch.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
