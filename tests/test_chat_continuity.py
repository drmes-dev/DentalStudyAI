"""Regression coverage for the actual MCQ conversation failures."""
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import PropertyMock, patch

for name in ('GEMINI_API_KEY', 'GROQ_API_KEY', 'DASHSCOPE_API_KEY'):
    os.environ.setdefault(name, 'offline-test-key')
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'backend'))
import main
from conversation import MCQTokens, answer_label, validate_question

QUOTE = "The Adams clasp provides retention for removable orthodontic appliances."
SOURCE = {"text": QUOTE, "filename": "orthodontics.pdf", "page": 10,
          "evidence_kind": "exposition", "score": 0.9}
QUESTION = {"stem": "Which component provides retention in a removable orthodontic appliance?",
            "options": {"A": "Spring", "B": "Adams clasp", "C": "Labial bow", "D": "Expansion screw"},
            "correct_answer": "B", "explanation": QUOTE + " [S1]",
            "support": [{"label": "S1", "quote": QUOTE}]}


class ChatContinuityTests(unittest.TestCase):
    def setUp(self):
        for p in [patch.object(main, 'require_beta_access'),
                  patch.object(main, 'enforce_rate_limit'),
                  patch.object(type(main.rag_store), 'configured', new_callable=PropertyMock, return_value=True)]:
            p.start()
            self.addCleanup(p.stop)
        self.search = patch.object(main.rag_store, 'search', return_value=[SOURCE]).start()
        self.addCleanup(patch.stopall)

    def request(self, message, **kwargs):
        return main.ChatRequest(message=message, mode='mcq', session_id='student', chat_id='chat-one', **kwargs)

    def generate(self):
        results = [{"provider": "Mock", "response": json.dumps(QUESTION)},
                   {"provider": "Mock", "response": '{"verified": true, "correct_answer": "B"}'}]
        with patch.object(main, 'generate_with_fallback', side_effect=results):
            return main.chat(self.request('Give me one easy BDS-level MCQ on orthodontics. Ask one question at a time.'), None)

    def test_question_waits_for_answer_without_leaking_key_or_evidence(self):
        result = self.generate()
        self.assertIn('B.** Adams clasp', result['response'])
        self.assertNotIn('correct_answer', result['response'])
        self.assertNotIn('explanation', result['response'])
        self.assertNotIn(QUOTE, result['response'])
        self.assertEqual(result['sources'], [])
        self.assertTrue(result['grounding_verified'])
        self.assertEqual(self.search.call_args.kwargs['exclude_assessment'], True)

    def test_B_and_B_with_text_grade_the_original_key_without_AI_or_RAG(self):
        token = self.generate()['mcq_token']
        self.search.reset_mock()
        for reply in ['B', "B. Adams clasp", 'option b', 'Adams clasp']:
            with patch.object(main, 'generate_with_fallback') as ai:
                result = main.chat(self.request(reply, mcq_token=token), None)
            self.assertIn('**Correct.**', result['response'])
            self.assertIn('**B. Adams clasp**', result['response'])
            self.assertNotIn('option a', result['response'])
            ai.assert_not_called()
        self.search.assert_not_called()

    def test_wrong_answer_and_why_keep_B(self):
        token = self.generate()['mcq_token']
        result = main.chat(self.request('A', mcq_token=token), None)
        self.assertIn('The correct answer is B.', result['response'])
        with patch.object(main, 'generate_with_fallback') as ai:
            why = main.chat(self.request('why', mcq_token=token), None)
        self.assertIn('Answer: B. Adams clasp', why['response'])
        ai.assert_not_called()

    def test_next_preserves_subject_and_avoids_previous_stem(self):
        token = self.generate()['mcq_token']
        with patch.object(main, 'generate_chat_mcq', return_value={}) as generate:
            main.chat(self.request('Next question', mcq_token=token), None)
        self.assertIn('orthodontics', self.search.call_args.args[0])
        self.assertIn('orthodontics', generate.call_args.args[2])
        self.assertEqual(generate.call_args.args[3]['seen'], [QUESTION['stem']])

    def test_tampered_cross_chat_and_cross_session_tokens_are_rejected(self):
        cipher = MCQTokens('stable-server-secret')
        token = cipher.seal({'question': QUESTION}, 'student', 'chat-one')
        self.assertIsNone(cipher.open(token, 'student', 'chat-two'))
        self.assertIsNone(cipher.open(token, 'someone-else', 'chat-one'))
        self.assertIsNone(cipher.open(token[:-5] + 'abcde', 'student', 'chat-one'))
        # Persisted state can be reopened after a server restart.
        self.assertEqual(MCQTokens('stable-server-secret').open(token, 'student', 'chat-one')['question'], QUESTION)

    def test_old_question_has_no_invented_answer(self):
        with patch.object(main, 'generate_with_fallback') as ai:
            result = main.chat(self.request('B'), None)
        self.assertIn('no saved, verified answer key', result['response'])
        ai.assert_not_called()

    def test_independent_review_rejects_changed_labels_or_uncertainty(self):
        for verdict in [{'verified': True, 'correct_answer': 'A'}, {'verified': False, 'correct_answer': 'B'}]:
            with patch.object(main, 'generate_with_fallback', side_effect=[
                    {'provider': 'Mock', 'response': json.dumps(QUESTION)},
                    {'provider': 'Mock', 'response': json.dumps(verdict)}]):
                result = main.chat(self.request('Give me an orthodontics MCQ'), None)
            self.assertFalse(result['grounding_verified'])
            self.assertFalse(result['mcq_token'])
            self.assertNotIn(QUESTION['stem'], result['response'])

    def test_fabricated_quotes_and_exam_distractors_are_rejected(self):
        bad = {**QUESTION, 'support': [{'label': 'S1', 'quote': 'Completely fabricated textbook quotation about active clasps.'}]}
        with self.assertRaises(ValueError):
            validate_question(bad, [SOURCE])
        with self.assertRaises(ValueError):
            validate_question(QUESTION, [{**SOURCE, 'evidence_kind': 'assessment'}])

    def test_short_followup_uses_history_for_search_and_prompt(self):
        history = [main.ChatTurn(role='user', content='Explain retention in removable orthodontic appliances for my viva'),
                   main.ChatTurn(role='assistant', content='We were discussing the Adams clasp.')]
        with patch.object(main, 'generate_with_fallback', return_value={'provider': 'Mock', 'response': 'Draft'}) as ai:
            with patch.object(main, 'verify_source_grounding', return_value={'provider': 'Mock', 'response': 'Checked'}):
                main.chat(main.ChatRequest(message='Why?', history=history), None)
        self.assertIn('orthodontic', self.search.call_args.args[0])
        self.assertIn('Adams clasp', ai.call_args.args[0])

    def test_failed_source_verification_withholds_draft(self):
        with patch.object(main, 'generate_with_fallback', return_value={'provider': 'Mock', 'response': 'Unverified factual answer'}):
            with patch.object(main, 'verify_source_grounding', return_value={'provider': 'Error', 'response': 'Quota exhausted'}):
                result = main.chat(main.ChatRequest(message='Explain clasp retention'), None)
        self.assertNotIn('Unverified factual answer', result['response'])
        self.assertFalse(result['grounding_verified'])


if __name__ == '__main__':
    unittest.main()
