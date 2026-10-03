"""Regression tests for import failures, immutable tests and PDF mode isolation."""
import os
import sys
import unittest
import asyncio
import threading
from io import BytesIO
from pathlib import Path
from unittest.mock import patch, PropertyMock, Mock
from fastapi import HTTPException
from starlette.datastructures import UploadFile

for key in ('GEMINI_API_KEY', 'GROQ_API_KEY', 'DASHSCOPE_API_KEY'):
    os.environ.setdefault(key, 'offline-test-key')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import main
from test_engine import PastPaperStore, numeric_metadata, _json


class ModeStabilityTests(unittest.TestCase):
    def setUp(self):
        for name in ('require_beta_access', 'enforce_rate_limit'):
            item = patch.object(main, name)
            item.start()
            self.addCleanup(item.stop)

    def question(self):
        return dict(id='q1', stem='Which option is supported?', options={'A': 'One', 'B': 'Two'},
                    paper_id='paper', paper_title='Exam', subject='Ortho', year='2025',
                    question_number='1', topic='Retention', subtopic='', page=1, stem_hash='stem',
                    verified_answer='B', provided_answer='', provisional_answer='',
                    verification_status='verified', verification_confidence='high',
                    verification_rationale='Textbook support', verification_sources=[])

    def test_numeric_metadata_handles_production_failures_without_guessing_marks(self):
        self.assertEqual(numeric_metadata('267-268', integer=True), 267)
        self.assertEqual(numeric_metadata('267–268', integer=True), 267)
        self.assertEqual(numeric_metadata('2.5 marks'), 2.5)
        for value in ('Unknown', '?', None, 'NaN', float('inf'), -2, True, {}, '2 or 5'):
            self.assertEqual(numeric_metadata(value), 0)
            self.assertEqual(numeric_metadata(value, integer=True), 0)

    def test_malformed_replacement_does_not_delete_usable_paper(self):
        store = PastPaperStore()
        index = Mock()
        store._index = lambda: index
        with patch.object(store, '_delete_prefix') as delete:
            with self.assertRaises(ValueError):
                store.index_paper(paper_id='paper', title='Exam', subject='Ortho', year='2025',
                                  filename='exam.pdf', questions=[None, {'stem': ''}])
        delete.assert_not_called()
        index.delete.assert_not_called()
        index.upsert.assert_not_called()

    def test_unknown_marks_and_page_ranges_can_be_saved(self):
        store = PastPaperStore()
        index = Mock()
        store._index = lambda: index
        store.index_paper(paper_id='paper', title='Exam', subject='Ortho', year='2025', filename='exam.pdf',
                         replace_existing=False, questions=[None, {'stem': 'Actual source question',
                         'options': {'A': 'One', 'B': 'Two'}, 'provided_answer': 'Z',
                         'suggested_answer': 'B', 'marks': 'Unknown', 'page': '267-268'}])
        metadata = index.upsert.call_args_list[0].kwargs['vectors'][0]['metadata']
        self.assertEqual(metadata['marks'], 0)
        self.assertEqual(metadata['page'], 267)
        self.assertEqual(metadata['provided_answer'], '')
        self.assertEqual(metadata['provisional_answer'], 'B')

    def test_extended_matching_options_i_and_j_are_preserved(self):
        store = PastPaperStore()
        index = Mock()
        store._index = lambda: index
        for options in ({chr(65+i): f'Option {i}' for i in range(10)}, [f'Option {i}' for i in range(10)]):
            store.index_paper(paper_id='paper', title='Exam', subject='Ortho', year='2025', filename='exam.pdf',
                             replace_existing=False, questions=[{'stem':'An extended matching question',
                             'options':options,'provided_answer':'I'}])
            metadata = index.upsert.call_args_list[-2].kwargs['vectors'][0]['metadata']
            self.assertIn('J', __import__('json').loads(metadata['options_json']))
            self.assertEqual(metadata['provided_answer'], 'I')

    def test_serialization_never_silently_truncates_json(self):
        with self.assertRaises(ValueError):
            _json({'A': 'x' * 13000})

    def test_invalid_stored_key_is_excluded(self):
        store = PastPaperStore()
        item = store._metadata_to_question('q', {'options_json': '{"A":"One","B":"Two"}',
                      'provided_answer': 'Z', 'page': '?', 'marks': 'Unknown'})
        self.assertEqual(item['provided_answer'], '')
        with patch.object(store, 'fetch_questions', return_value=[item]):
            with self.assertRaisesRegex(ValueError, 'unresolved'):
                store.grade([{'question_id': 'q', 'answer': 'A'}])

    def test_snapshot_grades_without_bank_or_ai_even_after_key_changes(self):
        original = self.question()
        public = {key: value for key, value in original.items() if key not in
                  ('verified_answer', 'provided_answer', 'provisional_answer')}
        with patch.object(main.past_paper_store, 'sample_test', return_value=[public]), \
             patch.object(main.past_paper_store, 'fetch_questions', return_value=[original]):
            result = main.start_test(main.TestStartRequest(session_id='visitor'), None, None)
        self.assertNotIn('verified_answer', result['questions'][0])
        original['verified_answer'] = 'A'
        with patch.object(main.past_paper_store, 'fetch_questions', side_effect=AssertionError('No bank read')), \
             patch.object(main.past_paper_store, 'list_questions', side_effect=AssertionError('No bank read')), \
             patch.object(main, 'generate_with_fallback', side_effect=AssertionError('No AI call')):
            grade = main.grade_test(main.TestGradeRequest(session_id='visitor', test_token=result['test_token'],
                                    responses=[{'question_id': 'q1', 'answer': 'B'}]), None, None)
        self.assertEqual(grade['score_percent'], 100)
        self.assertEqual(grade['details'][0]['correct_answer'], 'B')
        with self.assertRaises(HTTPException) as exc:
            main.grade_test(main.TestGradeRequest(session_id='other', test_token=result['test_token']), None, None)
        self.assertEqual(exc.exception.status_code, 409)

    def test_unanswered_snapshot_questions_still_count_and_unknown_ids_fail(self):
        store = PastPaperStore()
        grade = store.grade([], questions=[self.question()])
        self.assertEqual(grade['total'], 1)
        self.assertEqual(grade['unanswered'], 1)
        with self.assertRaises(ValueError):
            store.grade([{'question_id': 'different', 'answer': 'B'}], questions=[self.question()])

    def test_pdf_tutor_requires_its_session_pdf_without_ai_or_library_fallback(self):
        with patch.dict(main.session_pdfs, {}, clear=True), \
             patch.object(main.rag_store, 'search', side_effect=AssertionError('No other books')), \
             patch.object(main, 'generate_with_fallback', side_effect=AssertionError('No AI')):
            result = main.chat(main.ChatRequest(message='Explain this PDF', mode='pdf', session_id='fresh'), None, None)
        self.assertTrue(result['pdf_unavailable'])
        self.assertEqual(result['sources'], [])

    def test_pdf_tutor_uses_only_uploaded_pdf(self):
        source = {'name': 'fixture.pdf', 'uploaded_at': main.time.time(), 'chunks': [{'page': 1, 'text': 'This fixture contains explanatory textbook prose.', 'chunk_index': 0}]}
        with patch.dict(main.session_pdfs, {'fixture': source}, clear=True), \
             patch.object(type(main.rag_store), 'configured', new_callable=PropertyMock, return_value=True), \
             patch.object(main.rag_store, 'search', side_effect=AssertionError('No other books')), \
             patch.object(main, 'generate_with_fallback', return_value={'provider': 'Mock', 'response': 'Supported [S1]'}), \
             patch.object(main, 'verify_source_grounding', return_value={'provider': 'Mock', 'response': 'Supported [S1]'}):
            result = main.chat(main.ChatRequest(message='Explain the fixture', mode='pdf', session_id='fixture'), None, None)
        self.assertEqual([s['filename'] for s in result['sources']], ['fixture.pdf'])

    def test_provider_fallback_has_total_deadline(self):
        gemini, groq = Mock(), Mock()
        gemini.models.generate_content.side_effect = RuntimeError('Timeout')
        with patch.object(main, 'gemini_client', gemini), patch.object(main, 'groq_client', groq), \
             patch.object(main, 'model_availability', main.ModelAvailability()), \
             patch.object(main.time, 'monotonic', side_effect=[0, 0, 46, 46]):
            result = main.generate_with_fallback('Question', timeout_seconds=45)
        self.assertEqual(result['provider'], 'Error')
        groq.with_options.assert_not_called()

    def test_bare_topic_in_mcq_mode_starts_a_question(self):
        with patch.object(type(main.rag_store), 'configured', new_callable=PropertyMock, return_value=False), \
             patch.object(main, 'generate_chat_mcq', return_value={'response': 'Question'}) as generate:
            result = main.chat(main.ChatRequest(message='Orthodontics', mode='mcq'), None, None)
        self.assertEqual(result['response'], 'Question')
        generate.assert_called_once()

    def test_pdf_extraction_does_not_block_other_requests(self):
        started = threading.Event()
        release = threading.Event()
        def extract(*args, **kwargs):
            started.set()
            release.wait(5)
            return {'pages': [], 'page_count': 1, 'unresolved_scanned_pages': [1]}
        async def exercise():
            task = asyncio.create_task(main.upload_pdf(None, UploadFile(BytesIO(b'fixture'), filename='fixture.pdf'), 'fixture-session', None))
            try:
                for _ in range(100):
                    if started.is_set():
                        break
                    await asyncio.sleep(0.01)
                self.assertTrue(started.is_set())
                self.assertFalse(release.is_set())
                self.assertTrue(main.health()['ok'])
            finally:
                release.set()
            result = await task
            self.assertFalse(result['success'])
        with patch.object(main, 'PdfReader', return_value=Mock(pages=[1])), \
             patch.object(main, 'extract_pdf_pages', side_effect=extract):
            asyncio.run(exercise())


if __name__ == '__main__':
    unittest.main()
