"""Exercise the automatic RAG import, including the real upload category."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, PropertyMock

for name in ('GEMINI_API_KEY', 'GROQ_API_KEY', 'DASHSCOPE_API_KEY'):
    os.environ.setdefault(name, 'offline-test-key')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import main


class PastPaperImportTests(unittest.TestCase):
    def setUp(self):
        self.documents = [{
            'doc_id': 'scanned', 'category': 'Past Papers',
            'filename': 'scan.pdf', 'title': 'Scan', 'pages': 3,
        }]
        self.patches = [
            patch.object(main, 'require_owner_access'),
            patch.object(main, 'require_beta_access'),
            patch.object(main, 'enforce_rate_limit'),
            patch.object(type(main.rag_store), 'configured', new_callable=PropertyMock, return_value=True),
            patch.object(main.rag_store, 'list_documents', return_value=self.documents),
            patch.object(main.past_paper_store, 'list_papers', return_value=[]),
        ]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def test_plural_upload_category_imports_without_content_detection(self):
        for category in ('Past Papers', 'Past Paper', 'PAST PAPERS'):
            self.documents[0]['category'] = category
            with patch.object(main, 'structure_existing_rag_paper', return_value={'success': True, 'imported': True}) as structure:
                with patch.object(main.rag_store, 'document_sample_pages') as sample:
                    result = main.import_existing_past_paper(None, None)
            self.assertTrue(result['imported'])
            structure.assert_called_once_with(self.documents[0])
            sample.assert_not_called()

    def test_plural_category_is_counted_as_unsynced_in_catalog(self):
        with patch.object(main.past_paper_store, 'catalog', return_value={'paper_count': 0, 'papers': []}):
            result = main.test_catalog(None, None)
        self.assertEqual(result['rag_past_paper_count'], 1)
        self.assertFalse(result['rag_sync_complete'])

    def test_import_failure_is_not_success_or_empty_library(self):
        with patch.object(main, 'structure_existing_rag_paper', side_effect=ValueError('No readable OCR text')):
            result = main.import_existing_past_paper(None, None)
        self.assertFalse(result['success'])
        self.assertFalse(result['imported'])
        self.assertIn('No readable OCR text', result['message'])
        self.assertEqual(result['failures'][0]['doc_id'], 'scanned')

    def test_already_imported_paper_is_not_reprocessed(self):
        with patch.object(main.past_paper_store, 'list_papers', return_value=[{'paper_id': 'scanned'}]):
            with patch.object(main, 'structure_existing_rag_paper') as structure:
                result = main.import_existing_past_paper(None, None)
        self.assertTrue(result['success'])
        self.assertFalse(result['imported'])
        structure.assert_not_called()


if __name__ == '__main__':
    unittest.main()
