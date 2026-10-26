import unittest
from publication_language import editorial_history_hits, publication_terms


class PublicationLanguageTests(unittest.TestCase):
    def test_revision_history_is_rejected_in_reader_facing_prose(self):
        for text in ['The revised experiments use new models.',
                     'Historical scores precede the revision.',
                     'Earlier results were corrected after quarantine.',
                     'The author review is a remaining submission task.']:
            self.assertTrue(editorial_history_hits(text), text)

    def test_artifact_paths_and_internal_labels_keep_their_identities(self):
        tex = r'\label{sec:revised-data}\path{analysis/revision_v1/protocol.json}\input{tables/table_revision_archived_ci.tex}'
        self.assertFalse(editorial_history_hits(tex))
        self.assertEqual(publication_terms(tex), tex)
        code = '\\begin{Verbatim}\npython scripts/revision/replay_reviewer_results.py\n\\end{Verbatim}'
        self.assertFalse(editorial_history_hits(code))
        self.assertTrue(editorial_history_hits(code+' Historical results.'))

    def test_model_adaptation_and_token_measurement_are_scientific_conditions(self):
        self.assertFalse(editorial_history_hits(
            'The same model is scored before and after adaptation with 32 new tokens.'))

    def test_benchmark_display_names_do_not_rename_frozen_evidence(self):
        tex = r'Legacy views; legacy & 0.8 \\ \label{sup:legacy}\path{dataset/legacy_subsets/}'
        actual = publication_terms(tex)
        self.assertIn('Benchmark views; benchmark &', actual)
        self.assertIn(r'\label{sup:legacy}', actual)
        self.assertIn(r'\path{dataset/legacy_subsets/}', actual)


if __name__ == '__main__':
    unittest.main(verbosity=2)
