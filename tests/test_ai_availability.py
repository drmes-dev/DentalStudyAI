import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

for name in ('GEMINI_API_KEY', 'GROQ_API_KEY', 'DASHSCOPE_API_KEY'):
    os.environ.setdefault(name, 'offline-test-key')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import main
from ai_availability import AIUnavailable, ModelAvailability


def api_error(status, message, retry_after=None):
    error = RuntimeError(message)
    error.status_code = status
    error.response = SimpleNamespace(headers={'retry-after': str(retry_after)} if retry_after else {})
    return error


def completion(content, finish_reason='stop'):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)])


class AIAvailabilityTests(unittest.TestCase):
    def test_compact_fallback_switches_provider_and_skips_exhausted_models(self):
        gemini, groq, qwen = Mock(), Mock(), Mock()
        groq.with_options.return_value = groq
        qwen.with_options.return_value = qwen
        gemini.models.generate_content.side_effect = api_error(429, 'GenerateRequestsPerDayPerProjectPerModel-FreeTier')
        groq.chat.completions.create.side_effect = api_error(429, 'TPD: try again in 22m5.5s')
        qwen.chat.completions.create.return_value = completion('{"questions":[]}')
        with patch.object(main, 'gemini_client', gemini), patch.object(main, 'groq_client', groq), \
             patch.object(main, 'qwen_client', qwen), patch.object(main, 'model_availability', ModelAvailability()):
            first = main.generate_with_fallback('Return JSON', timeout_seconds=60, json_response=True)
            second = main.generate_with_fallback('Return JSON', timeout_seconds=60, json_response=True)
        self.assertEqual(first['model'], 'qwen-flash')
        self.assertEqual(second['model'], first['model'])
        self.assertEqual(gemini.models.generate_content.call_count, 1)
        self.assertEqual([call.kwargs['model'] for call in groq.chat.completions.create.call_args_list],
                         ['qwen/qwen3.8-27b'])
        self.assertEqual([call.kwargs['model'] for call in qwen.chat.completions.create.call_args_list],
                         ['qwen-flash', 'qwen-flash'])

    def test_qwen_is_enabled_for_json_using_existing_groq_key(self):
        gemini, groq = Mock(), Mock()
        gemini.models.generate_content.side_effect = api_error(429, 'Daily quota')
        groq.with_options.return_value = groq
        groq.chat.completions.create.return_value = completion('{"questions":[]}')
        with patch.object(main, 'gemini_client', gemini), patch.object(main, 'groq_client', groq), \
             patch.object(main, 'model_availability', ModelAvailability()):
            result = main.generate_with_fallback('JSON', timeout_seconds=60, json_response=True)
        self.assertEqual(result['model'], 'qwen/qwen3.8-27b')
        self.assertEqual(result['provider'], 'Groq')

    def test_daily_gemini_quota_waits_until_pacific_midnight_not_retryinfo(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime(2026, 9, 30, 9, 0, tzinfo=ZoneInfo('America/Los_Angeles')).timestamp()
        availability = ModelAvailability()
        with patch('ai_availability.time.time', return_value=now):
            availability.failed('Gemini', 'model', api_error(429, 'GenerateRequestsPerDayPerProjectPerModel', 47))
            self.assertEqual(availability.remaining('Gemini', 'model'), 15 * 3600 + 5)

    def test_retry_header_duration_and_denied_access_are_sanitized(self):
        availability = ModelAvailability()
        with patch('ai_availability.time.time', return_value=100):
            availability.failed('Groq', 'model', api_error(429, 'secret: try again in 22m5.5s', 60))
            availability.failed('Qwen API', 'model', api_error(403, 'secret AccessDenied.Unpurchased'))
            result = availability.unavailable([('Groq', 'model'), ('Qwen API', 'model')])
            self.assertEqual(result['retry_after'], 1326)
            self.assertEqual(result['error_code'], 'quota_reached')
            self.assertNotIn('secret', str(result))
            self.assertEqual(availability.remaining('Qwen API', 'model'), 3600)

    def test_truncated_json_falls_back_instead_of_saving_partial_questions(self):
        gemini, groq, qwen = Mock(), Mock(), Mock()
        gemini.models.generate_content.side_effect = api_error(403, 'Denied')
        groq.with_options.return_value = groq
        qwen.with_options.return_value = qwen
        groq.chat.completions.create.return_value = completion('{"questions":[]}', 'length')
        qwen.chat.completions.create.return_value = completion('{"questions":[]}')
        with patch.object(main, 'gemini_client', gemini), patch.object(main, 'groq_client', groq), \
             patch.object(main, 'qwen_client', qwen), patch.object(main, 'model_availability', ModelAvailability()):
            result = main.generate_with_fallback('JSON', timeout_seconds=60, json_response=True)
        self.assertEqual(result['model'], 'qwen-flash')

    def test_quota_failure_does_not_advance_or_write_page_checkpoint(self):
        result = {'provider': 'Error', 'response': 'Free AI quota reached', 'retry_after': 900, 'error_code': 'quota_reached'}
        document = {'doc_id': 'scanned', 'filename': 'Ortho.pdf', 'title': 'Ortho Past Paper'}
        with patch.object(main.rag_store, 'document_pages', return_value=[{'page': 1, 'text': 'Exam MCQs'}]), \
             patch.object(main, 'generate_with_fallback', return_value=result), \
             patch.object(main.past_paper_store, 'index_paper') as save:
            with self.assertRaises(AIUnavailable):
                main.structure_existing_rag_paper(document, incremental=True)
        save.assert_not_called()

    def test_worker_waits_for_quota_then_resumes_to_completion(self):
        clock = [100.0]
        def sleep(seconds):
            self.assertLessEqual(seconds, 60)
            clock[0] += seconds
        results = [{'success': False, 'retry_after': 130, 'message': 'Free AI quota reached'},
                   {'success': True, 'imported': True, 'processed_pages': 4, 'total_pages': 4},
                   {'success': True, 'imported': False}]
        state = {'running': True, 'saved_batches': 0}
        with patch.object(main, '_paper_sync_state', state), \
             patch.object(main, 'import_next_existing_paper', side_effect=results), \
             patch.object(main.time, 'time', side_effect=lambda: clock[0]), patch.object(main.time, 'sleep', side_effect=sleep):
            main.paper_sync_worker()
        self.assertTrue(state['complete'])
        self.assertEqual(state['saved_batches'], 1)
        self.assertFalse(state['running'])
        self.assertIsNone(state['retry_at'])

    def test_answer_review_defers_without_spending_ai_quota_during_import(self):
        with patch.object(main, 'require_owner_access'), patch.object(main, 'enforce_rate_limit'), \
             patch.object(main, '_paper_sync_state', {'running': True}), \
             patch.object(main.past_paper_store, 'pending_questions') as pending:
            result = main.verify_past_paper('paper', None)
        self.assertTrue(result['deferred'])
        pending.assert_not_called()
