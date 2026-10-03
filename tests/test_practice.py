"""Interactive viva/OSCE must ask first and retain a source-checked reference."""
import os
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch, PropertyMock

for key in ('GEMINI_API_KEY', 'GROQ_API_KEY', 'DASHSCOPE_API_KEY'):
    os.environ.setdefault(key, 'offline-test-key')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import main
from practice import generate_practice, validate_practice


class PracticeTests(unittest.TestCase):
    def setUp(self):
        self.quote = 'The fixture reference says that the retention component maintains appliance position.'
        self.sources = [{'text': self.quote, 'filename': 'fixture.pdf', 'page': 4, 'evidence_kind': 'exposition'}]
        self.candidate = {'question': 'What is the function of the retention component?',
                          'expected_answer': 'It maintains appliance position. [S1]',
                          'support': [{'label': 'S1', 'quote': self.quote}]}
        for item in [patch.object(main, 'require_beta_access'), patch.object(main, 'enforce_rate_limit'),
                     patch.object(type(main.rag_store), 'configured', new_callable=PropertyMock, return_value=True),
                     patch.object(main.rag_store, 'search', return_value=self.sources)]:
            item.start()
            self.addCleanup(item.stop)

    def start(self, mode='viva'):
        request = main.ChatRequest(message=f'Start {mode} practice on appliance retention', mode=mode,
                                   session_id='student', chat_id='practice')
        with patch.object(main, 'generate_with_fallback', side_effect=[
            {'provider': 'Mock', 'response': json.dumps(self.candidate)},
            {'provider': 'Mock', 'response': '{"verified":true}'}]):
            return request, main.chat(request, None, None)

    def test_viva_asks_exactly_one_question_and_hides_the_reference(self):
        request, result = self.start()
        self.assertIn(self.candidate['question'], result['response'])
        self.assertNotIn(self.candidate['expected_answer'], result['response'])
        self.assertEqual(result['response'].count('?'), 1)
        self.assertEqual(result['sources'], [])
        state = main.practice_tokens.open(result['practice_token'], request.session_id, request.chat_id)
        self.assertEqual(state['expected_answer'], self.candidate['expected_answer'])
        self.assertIsNone(main.practice_tokens.open(result['practice_token'], request.session_id, 'another-chat'))

    def test_osce_hides_its_reference_until_requested(self):
        self.candidate['question'] = 'Station: inspect a removable appliance. Task: explain its retention component.'
        request, result = self.start('osce')
        self.assertIn('OSCE station', result['response'])
        self.assertNotIn(self.candidate['expected_answer'], result['response'])
        with patch.object(main, 'generate_with_fallback', side_effect=AssertionError('Saved reference is deterministic')):
            shown = main.chat(request.model_copy(update={'message': 'show answer', 'practice_token': result['practice_token']}), None, None)
        self.assertEqual(shown['response'], self.candidate['expected_answer'])

    def test_feedback_preserves_the_saved_question_and_references(self):
        request, result = self.start()
        with patch.object(main.rag_store, 'search', side_effect=AssertionError('Do not retrieve a different question')), \
             patch.object(main, 'generate_with_fallback', side_effect=[
                {'provider': 'Mock', 'response': '{"feedback":"Correct: it maintains appliance position. [S1]"}'},
                {'provider': 'Mock', 'response': '{"verified":true}'}]) as generate:
            grade = main.chat(request.model_copy(update={'message':'It maintains position', 'practice_token': result['practice_token']}), None, None)
        self.assertIn(self.candidate['question'], generate.call_args_list[0].args[0])
        self.assertIn('**Feedback**', grade['response'])
        self.assertEqual(grade['practice_token'], result['practice_token'])
        self.assertEqual(grade['sources'][0]['page'], 4)

    def test_next_retains_subject_and_withholds_unverified_feedback(self):
        request, result = self.start()
        with patch.object(main, 'generate_practice', return_value={'response':'Next'}) as generate:
            main.chat(request.model_copy(update={'message':'next', 'practice_token':result['practice_token']}), None, None)
        self.assertEqual(generate.call_args.args[2], request.message)
        with patch.object(main, 'generate_with_fallback', side_effect=[
            {'provider':'Mock', 'response':'{"feedback":"Invented detail"}'},
            {'provider':'Mock', 'response':'{"verified":false}'}]):
            response = main.chat(request.model_copy(update={'message':'my answer', 'practice_token':result['practice_token']}), None, None)
        self.assertNotIn('Invented detail', response['response'])
        self.assertEqual(response['practice_token'], result['practice_token'])

    def test_question_format_and_invented_evidence_are_rejected(self):
        for changes in ({'question':'Here is the model answer without a question'},
                        {'question':'First question? Second question?'},
                        {'support':[{'label':'S1','quote':'An invented quotation that does not appear in the source.'}]}):
            with self.assertRaises(ValueError):
                validate_practice({**self.candidate, **changes}, self.sources, 'viva')


if __name__ == '__main__':
    unittest.main()
