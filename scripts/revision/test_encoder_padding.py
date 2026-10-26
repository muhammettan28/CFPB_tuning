import unittest
from run_revision_gpu import encoder_inputs


class StrictTokenizer:
    """Model the tokenizer's conflicting simultaneous truncation/padding guard."""
    def __call__(self,texts,truncation,max_length,padding=False,pad_to_multiple_of=None,**kwargs):
        if truncation and padding and pad_to_multiple_of and max_length%pad_to_multiple_of:
            raise ValueError('Truncation and padding lengths are incompatible')
        return {'input_ids':[list(range(min(len(t),max_length))) for t in texts],
                'attention_mask':[[1]*min(len(t),max_length) for t in texts]}

    def pad(self,encoded,padding,pad_to_multiple_of,return_tensors):
        self.encoding_before_padding=encoded
        size=max(map(len,encoded['input_ids']))
        size=((size+pad_to_multiple_of-1)//pad_to_multiple_of)*pad_to_multiple_of
        return {k:[v+[0]*(size-len(v)) for v in values] for k,values in encoded.items()}


class EncoderPaddingTests(unittest.TestCase):
    def test_cap_100_keeps_100_real_tokens_and_pads_to_104(self):
        tok=StrictTokenizer();inputs=encoder_inputs(tok,['x'*150,'y'*12],100)
        self.assertEqual([sum(m) for m in inputs['attention_mask']],[100,12])
        self.assertEqual([len(x) for x in inputs['input_ids']],[104,104])
        self.assertEqual([len(x) for x in tok.encoding_before_padding['input_ids']],[100,12])

    def test_aligned_caps_preserve_the_previous_inputs(self):
        for cap in [128,512]:
            tok=StrictTokenizer();texts=['x'*600,'y'*77]
            old=tok.pad(tok(texts,truncation=True,max_length=cap),True,8,'pt')
            self.assertEqual(encoder_inputs(tok,texts,cap),old)


if __name__=='__main__':unittest.main(verbosity=2)
