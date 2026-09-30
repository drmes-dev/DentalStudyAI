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
            structure.assert_called_once_with(self.documents[0], incremental=True, checkpoint=None)
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

    def test_partial_manifest_is_resumed(self):
        checkpoint = {'paper_id': 'scanned', 'import_complete': False, 'processed_pages': 2}
        with patch.object(main.past_paper_store, 'list_papers', return_value=[checkpoint]):
            with patch.object(main, 'structure_existing_rag_paper', return_value={'success': True, 'imported': True}) as structure:
                main.import_existing_past_paper(None, None)
        structure.assert_called_once_with(self.documents[0], incremental=True, checkpoint=checkpoint)

    def test_incremental_import_persists_two_pages_and_resumes_with_overlap(self):
        pages = [{'page': i, 'text': 'Exam text'} for i in range(1, 6)]
        question = {'stem': 'Actual source question?', 'options': {'A': 'one', 'B': 'two'}, 'page': 3}
        checkpoint = {'paper_id': 'scanned', 'processed_pages': 2, 'import_complete': False,
                      'subject': 'Orthodontics', 'year': '2025', 'title': 'Exam', 'question_count': 4}
        with patch.object(main.rag_store, 'document_pages', return_value=pages):
            with patch.object(main, 'parse_past_paper_questions', return_value=[question]) as parser:
                with patch.object(main.past_paper_store, 'index_paper', return_value={'success': True, 'question_count': 5}) as save:
                    result = main.structure_existing_rag_paper(self.documents[0], incremental=True, checkpoint=checkpoint)
        parser.assert_called_once_with(pages=pages[1:4], subject='Orthodontics')
        self.assertFalse(save.call_args.kwargs['replace_existing'])
        self.assertEqual(save.call_args.kwargs['import_progress']['processed_pages'], 4)
        self.assertFalse(result['complete'])
        self.assertEqual(result['processed_pages'], 4)

    def test_already_imported_paper_is_not_reprocessed(self):
        with patch.object(main.past_paper_store, 'list_papers', return_value=[{'paper_id': 'scanned'}]):
            with patch.object(main, 'structure_existing_rag_paper') as structure:
                result = main.import_existing_past_paper(None, None)
        self.assertTrue(result['success'])
        self.assertFalse(result['imported'])
        structure.assert_not_called()


if __name__ == '__main__':
    unittest.main()

class QuestionBankPipelineTests(unittest.TestCase):
    def test_ocr_question_saves_without_embedding_and_can_be_tested_and_graded(self):
        from test_engine import PastPaperStore
        records = {}
        class MemoryIndex:
            def upsert(self, vectors, namespace):
                for vector in vectors:
                    records[(namespace, vector['id'])] = vector
            def fetch(self, ids, namespace):
                return {'vectors': {key: records[(namespace, key)] for key in ids if (namespace, key) in records}}
        store = PastPaperStore()
        store._index = lambda: MemoryIndex()
        def list_ids(namespace, prefix):
            return [key for ns, key in records if ns == namespace and key.startswith(prefix)]
        with patch.object(main.rag_store, '_list_ids', side_effect=list_ids):
            with patch.object(main.rag_store, 'embed', side_effect=AssertionError('Bank must not require embeddings')):
                store.index_paper(paper_id='exam', title='Orthodontics Exam', subject='Orthodontics',
                    year='2025', filename='scan.pdf', replace_existing=False,
                    import_progress={'processed_pages': 2, 'import_complete': False},
                    questions=[{'stem': 'Which tooth is shown in the paper?', 'page': 2,
                                'options': {'A': 'Incisor', 'B': 'Canine'}, 'suggested_answer': 'B'}])
                self.assertEqual(store.catalog()['test_ready_questions'], 1)
                test = store.sample_test(count=20, subject='Orthodontics')
                self.assertEqual(len(test), 1)
                self.assertNotIn('provisional_answer', test[0])
                grade = store.grade([{'question_id': test[0]['id'], 'answer': 'B'}])
                self.assertEqual(grade['score_percent'], 100)
                # Retry is an idempotent upsert, not another copy of the MCQ.
                store.index_paper(paper_id='exam', title='Orthodontics Exam', subject='Orthodontics',
                    year='2025', filename='scan.pdf', replace_existing=False,
                    import_progress={'processed_pages': 2, 'import_complete': False},
                    questions=[{'stem': 'Which tooth is shown in the paper?', 'page': 2,
                                'options': {'A': 'Incisor', 'B': 'Canine'}, 'suggested_answer': 'B'}])
                self.assertEqual(store.catalog()['mcq_count'], 1)

class BackgroundImportTests(unittest.TestCase):
    def test_parser_requests_structured_json_and_preserves_source_question(self):
        import json
        raw = {'stem': 'Which tooth is shown?', 'options': {'A': 'Incisor', 'B': 'Canine'}, 'page': 2, 'suggested_answer': 'B'}
        with patch.object(main, 'generate_with_fallback', return_value={'provider': 'Groq', 'response': json.dumps({'questions': [raw]})}) as generate:
            result = main.parse_past_paper_questions(pages=[{'page': 2, 'text': 'Which tooth is shown? A. Incisor B. Canine'}], subject='Orthodontics')
        self.assertEqual(result, [raw])
        self.assertTrue(generate.call_args.kwargs['json_response'])

    def test_worker_retries_and_keeps_saved_progress(self):
        with main._paper_sync_lock:
            main._paper_sync_state.update(running=True, saved_batches=0, complete=False)
        results = [
            {'success': False, 'message': 'Temporary provider failure'},
            {'success': True, 'imported': True, 'title': 'Exam', 'processed_pages': 2, 'total_pages': 4, 'question_count': 10},
            {'success': True, 'imported': False},
        ]
        with patch.object(main, 'import_next_existing_paper', side_effect=results):
            with patch.object(main.time, 'sleep') as delay:
                main.paper_sync_worker()
        self.assertEqual([call.args[0] for call in delay.call_args_list], [15, 30])
        self.assertEqual(main._paper_sync_state['saved_batches'], 1)
        self.assertTrue(main._paper_sync_state['complete'])
        self.assertFalse(main._paper_sync_state['running'])
