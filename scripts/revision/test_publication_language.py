import unittest
from publication_language import editorial_history_hits, publication_terms, internal_workflow_hits


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

    def test_internal_delivery_notes_are_rejected_even_in_the_response_letter(self):
        for text in ['This is the local author-review version.',
                     'It has not been submitted.',
                     'No external access mechanism has been arranged.',
                     'The author must arrange access.',
                     'The evidence is maintained in a private repository.']:
            self.assertTrue(internal_workflow_hits(text), text)

    def test_obsolete_access_statements_and_template_prompts_are_rejected(self):
        for text in ['No reviewer/public upload was performed.',
                     'Manuscript integration/rendering and final author review remain separate.',
                     'Cleanup date: 26 October 2026.',
                     'The final delivery-readiness receipt is outside the ZIP.',
                     r'\section*{Funding, if any}',
                     r'\section*{Declaration of Generative AI and AI-assisted technologies, if any}']:
            self.assertTrue(internal_workflow_hits(text), text)

    def test_scientific_limitations_and_reviewer_change_explanations_are_allowed(self):
        self.assertFalse(internal_workflow_hits(
            'We added a public repository link. Upstream preprocessing is unavailable. '
            'The auxiliary fits lack complete fitted weights. '
            'AI-assisted tools supported wording and LaTeX formatting.'))


if __name__ == '__main__':
    unittest.main(verbosity=2)
