import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import analyze_corrected_gpu_results as analysis
from revision_gpu_common import LABELS,load_plan


class CorrectedOutputAnalysisTests(unittest.TestCase):
    def test_internal_alignment_uses_identity_and_checks_truth(self):
        target=pd.DataFrame({'record_id':[11,22],'true_label':LABELS[:2]})
        pred=pd.DataFrame({'record_id':['22','11'],'true_label':[LABELS[1],LABELS[0]],'prediction':[LABELS[1],LABELS[0]]})
        aligned=analysis.align_predictions(pred,target,'internal')
        self.assertEqual(aligned.record_id.tolist(),['11','22'])
        pred.loc[0,'true_label']=LABELS[2]
        with self.assertRaisesRegex(RuntimeError,'labels differ'):analysis.align_predictions(pred,target,'internal')

    def test_legacy_alignment_keeps_repeated_views_by_file_row(self):
        target=pd.DataFrame({'source_file':['a.csv','a.csv','b.csv'],'source_row':[0,1,0],'true_label':[LABELS[0],LABELS[1],LABELS[0]]})
        pred=target.iloc[[2,1,0]].copy();pred['prediction']=pred.true_label
        aligned=analysis.align_predictions(pred,target,'legacy')
        self.assertEqual(list(zip(aligned.source_file,aligned.source_row)),[('a.csv',0),('a.csv',1),('b.csv',0)])
        with self.assertRaisesRegex(RuntimeError,'view IDs differ'):
            analysis.align_predictions(pd.concat([pred,pred.iloc[:1]]),target,'legacy')

    def test_joint_group_bootstrap_identical_systems_have_zero_paired_difference(self):
        labels=LABELS+LABELS[:3]
        target=pd.DataFrame({'true_label':labels,'group':[0,0,1,2,3,4,5,6]})
        predictions=np.tile(np.asarray(labels)[:,None],(1,19))
        systems=[{'system':'qwen_base','indices':[0]},{'system':'qwen_ft','indices':[1,2,3]},
                 {'system':'llama_base','indices':[4]},{'system':'llama_ft','indices':[5,6,7]},
                 {'system':'bert','indices':[8,9,10]},{'system':'distilbert','indices':[11,12,13]},
                 {'system':'roberta','indices':[14,15,16]},{'system':'tfidf_10000','indices':[17]},
                 {'system':'tfidf_full','indices':[18]}]
        for s in systems:s.update(name=s['system'],train_rows=10000,stochastic_runs=3 if len(s['indices'])==3 else 0 if s['system'].startswith('tfidf') else 1)
        with tempfile.TemporaryDirectory() as tmp,patch.object(analysis,'OUT',Path(tmp)),patch.object(analysis,'REPLICATES',128),patch.object(analysis,'systems_and_predictions',return_value=(predictions,systems,[str(i) for i in range(19)])):
            rows,seeds,contrasts=analysis.summarise_role(target,'internal')
            self.assertTrue(all(abs(r['point']-1)<1e-12 for r in rows))
            self.assertTrue(all(r['paired_difference']==0 and r['low']==0 and r['high']==0 for r in contrasts))

    def test_latency_repeats_are_averaged_and_constant_paired_gap_is_exact(self):
        internal=pd.DataFrame({'record_id':[str(i) for i in range(100)],'group':np.repeat(np.arange(50),2)})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);prep=root/'prep';prep.mkdir()
            internal[['record_id']].to_csv(prep/'latency_sample_manifest.csv',index=False)
            for model in ['qwen','llama']:
                for variant in ['base','42','43','44']:
                    generation=2 if variant=='base' else {'42':3,'43':4,'44':5}[variant]
                    for condition in load_plan()['long_input']['conditions']:
                        out=root/'gpu'/'latency'/model/variant/condition['name'];out.mkdir(parents=True)
                        records=[{'record_id':str(i),'repetition':rep,'generation_seconds':generation,
                                  'end_to_end_seconds':generation+1,'input_tokens':100,'new_tokens':10}
                                 for rep in range(3) for i in range(100)]
                        pd.DataFrame(records).to_csv(out/'latency_records.csv',index=False)
            with patch.object(analysis,'PREP',prep),patch.object(analysis,'EXP',root/'gpu'),patch.object(analysis,'REPLICATES',128),patch.object(analysis,'completed',return_value={}),patch.object(analysis,'remember',return_value=None):
                summaries,paired=analysis.latency_results(internal)
                self.assertEqual(len(summaries),48)
                self.assertEqual(len(paired),12)
                for row in paired:
                    self.assertAlmostEqual(row['ft_seed_mean_minus_base_seconds'],2)
                    self.assertAlmostEqual(row['low'],2);self.assertAlmostEqual(row['high'],2)
                    self.assertEqual(row['groups'],50)


if __name__=='__main__':unittest.main(verbosity=2)
