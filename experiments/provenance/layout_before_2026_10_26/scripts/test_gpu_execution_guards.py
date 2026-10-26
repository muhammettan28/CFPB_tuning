import os
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import pandas as pd
import run_revision_gpu as runner
from types import SimpleNamespace


class LatencyIsolationTests(unittest.TestCase):
    def test_current_process_alone_is_allowed(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner.subprocess,'check_output',return_value=str(os.getpid())+'\n'):
            runner.require_no_other_compute(Path(tmp))
            self.assertFalse((Path(tmp)/'concurrent_compute_detected.json').exists())

    def test_other_compute_is_recorded_and_rejected_without_process_control(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(runner.subprocess,'check_output',return_value=str(os.getpid())+'\n987654321\n'):
            with self.assertRaisesRegex(RuntimeError,'Concurrent GPU compute'):
                runner.require_no_other_compute(Path(tmp))
            self.assertTrue((Path(tmp)/'concurrent_compute_detected.json').exists())


class EncoderValidationParityTests(unittest.TestCase):
    def test_reload_difference_blocks_final_inference_and_records_evidence(self):
        frame=pd.DataFrame({'record_id':[11,22],'product':runner.LABELS[:2],'narrative':['a','b']})
        args=SimpleNamespace(model='bert',variant='42')
        for expected,passed in [(runner.LABELS[:2],True),([runner.LABELS[0],runner.LABELS[0]],False)]:
            with self.subTest(passed=passed),tempfile.TemporaryDirectory(dir=runner.ROOT) as tmp:
                root=Path(tmp);source=root/'encoder/bert/seed_42';source.mkdir(parents=True)
                frame[['record_id','product']].assign(prediction=runner.LABELS[:2]).to_csv(source/'validation_predictions.csv',index=False)
                with patch.object(runner,'EXP',root),patch.object(runner,'load_frame',return_value=frame),patch.object(runner,'predict_encoder',return_value=(expected,[2,2])):
                    if passed:runner.require_encoder_validation_parity(args,{'encoder':{'eval_batch_size':16}},None,None,root)
                    else:
                        with self.assertRaisesRegex(RuntimeError,'final-test inference blocked'):
                            runner.require_encoder_validation_parity(args,{'encoder':{'eval_batch_size':16}},None,None,root)
                report=json.loads((root/'validation_inference_parity.json').read_text())
                self.assertEqual(report['passed'],passed)
                self.assertFalse(report['final_test_read_before_check'])


if __name__=='__main__':unittest.main(verbosity=2)
