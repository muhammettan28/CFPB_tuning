import tempfile
import unittest
from pathlib import Path
from prediction_cache import PredictionCache


class PredictionCacheTests(unittest.TestCase):
    def test_exact_inputs_reused_across_reopen_and_condition_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.sqlite3'
            identity={'checkpoint':'abc','generation':{'do_sample':False,'max_new_tokens':32}}
            tokens={'input_ids':[1,2,3],'attention_mask':[1,1,1]}
            cache=PredictionCache(path,identity)
            key=cache.key(tokens)
            self.assertIsNone(cache.get(key))
            value={'prediction':'credit_card','raw_response':'{"product":"credit_card"}','new_tokens':7}
            cache.put(key,value);cache.close()
            cache=PredictionCache(path,identity)
            self.assertEqual(cache.get(key),value)
            self.assertEqual(cache.hits,1);cache.close()

    def test_mask_checkpoint_and_generation_changes_invalidate_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'cache.sqlite3'
            original={'input_ids':[1,2],'attention_mask':[1,1]}
            masked={'input_ids':[1,2],'attention_mask':[1,0]}
            cache=PredictionCache(path,{'checkpoint':'a','max_new_tokens':32})
            cache.put(cache.key(original),{'prediction':'credit_card'})
            self.assertIsNone(cache.get(cache.key(masked)));cache.close()
            for identity in [{'checkpoint':'b','max_new_tokens':32},{'checkpoint':'a','max_new_tokens':16}]:
                cache=PredictionCache(path,identity)
                self.assertIsNone(cache.get(cache.key(original)));cache.close()

    def test_conflicting_response_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=PredictionCache(Path(tmp)/'cache.sqlite3',{'checkpoint':'a'})
            key=cache.key({'input_ids':[1],'attention_mask':[1]})
            cache.put(key,{'prediction':'credit_card'})
            with self.assertRaisesRegex(RuntimeError,'Conflicting'):
                cache.put(key,{'prediction':'debt_collection'})
            cache.close()


if __name__=='__main__':unittest.main(verbosity=2)
