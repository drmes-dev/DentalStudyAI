import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, PropertyMock
for name in ('GEMINI_API_KEY', 'GROQ_API_KEY', 'DASHSCOPE_API_KEY'):
    os.environ.setdefault(name, 'offline-test-key')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import main
from offline_pack import build_offline_pack


class OfflinePackTests(unittest.TestCase):
    def question(self, **values):
        return {'id': 'one', 'stem': 'Source question?', 'options': {'A': 'one', 'B': 'two'},
                'paper_id': 'paper', 'stem_hash': 'same', 'subject': 'Orthodontics', 'topic': 'Growth', **values}

    def test_preserves_answer_provenance_and_excludes_invalid_keys(self):
        source = [self.question(provided_answer='A'), self.question(id='two', verified_answer='B', verification_status='verified'),
                  self.question(id='three', provisional_answer='A'), self.question(id='four', provided_answer='Z'),
                  self.question(id='five', verified_answer='B', provided_answer='A', verification_status='conflict')]
        pack = build_offline_pack(source, [{'paper_id': 'paper', 'title': 'Exam'}])
        self.assertEqual(len(pack['questions']), 4)
        self.assertEqual([q['trusted_answer'] for q in pack['questions']], [True, True, False, False])
        self.assertEqual([q['answer_source'] for q in pack['questions']], ['past_paper_key', 'textbook_review', 'ai_prepared', 'textbook_review'])
        self.assertEqual(pack['questions'][0]['stem'], 'Source question?')
        self.assertEqual(pack['papers'][0]['title'], 'Exam')

    def test_caps_download_to_keep_device_storage_small(self):
        with self.assertRaisesRegex(ValueError, '8 MB'):
            build_offline_pack([self.question(stem='X' * (8 * 1024 * 1024), provided_answer='A')], [])

    def test_endpoint_requires_access_and_filters_the_requested_subject(self):
        with patch.object(main, 'require_beta_access') as access, patch.object(main, 'enforce_rate_limit'), \
             patch.object(type(main.past_paper_store), 'configured', new_callable=PropertyMock, return_value=True), \
             patch.object(main.past_paper_store, 'list_questions', return_value=[self.question(provided_answer='A')]) as questions, \
             patch.object(main.past_paper_store, 'list_papers', return_value=[]):
            result = main.offline_question_pack(None, 'Orthodontics', 'test-access')
        access.assert_called_once_with('test-access')
        questions.assert_called_once_with(subject='Orthodontics')
        self.assertEqual(result['schema_version'], 1)
