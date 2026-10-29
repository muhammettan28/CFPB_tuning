"""Guard against mixed manuscript versions and overwriting author corrections."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import refresh_response_locations as locations


class ResponseLocationTests(unittest.TestCase):
    def test_refresh_preserves_author_prose_and_is_idempotent_with_dual_pages(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp)/'response.tex'
            intro = 'All document locations refer to the clean manuscript and supplementary material.'
            source = intro+'\n'+''.join(
                '\\subsection*{'+ident+' — Title}\n'
                '\\begin{quote}Original reviewer comment\\end{quote}\n'
                '\\textbf{Response.} Author correction: retain 0.8402 and the manuscript wording.\n'
                '\\textbf{Changes and locations.} Manuscript Section 3.5 (p. 2).\n'
                for ident in [*(f'R1.{n}' for n in range(1, 9)),
                              *(f'R2.{n}' for n in range(1, 8))])
            path.write_text(source)
            fake_pages = lambda stem: ['', '3.5. Statistics'] if stem == 'manuscript' else ['', '', '', '3.5. Statistics']
            with patch.object(locations, 'REPORT', Path(tmp)/'locations.json'), \
                 patch.object(locations, 'ADDITIONS', {}), \
                 patch.object(locations, 'section_titles', return_value={'3.5': 'Statistics'}), \
                 patch.object(locations, 'pages', side_effect=fake_pages), \
                 patch.object(locations, 'sha', return_value='test-pdf-hash'):
                report = locations.refresh(path)
                first = path.read_text()
                self.assertEqual(first.count('(clean p. 2; tracked p. 4)'), 15)
                self.assertEqual(first.count('Author correction: retain 0.8402 and the manuscript wording.'), 15)
                locations.refresh(path)
                self.assertEqual(path.read_text(), first)
                self.assertEqual(locations.validate(report), 15)
                # Keeping the same cached file identity must not mask a wrong page.
                with patch.object(locations, 'pages', return_value=['3.5. Statistics']):
                    with self.assertRaises(AssertionError):
                        locations.validate(report)

    def test_repeated_or_missing_headings_cannot_be_certified(self):
        with patch.object(locations, 'section_titles', return_value={'4.3': 'Longer inputs'}):
            for pages in [[], ['4.3. Longer inputs', '4.3. Longer inputs']]:
                with patch.object(locations, 'pages', return_value=pages):
                    with self.assertRaises(ValueError):
                        locations.locate('tracked', 'Section', '4.3')


if __name__ == '__main__':
    unittest.main()
