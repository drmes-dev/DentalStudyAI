import unittest

from backend.main import is_strict_source_request


class StrictSourceRequestTests(unittest.TestCase):
    def test_detects_uploaded_textbook_request(self):
        self.assertTrue(
            is_strict_source_request(
                "According to the uploaded Endodontics textbook, explain EDTA."
            )
        )

    def test_detects_page_reference_request(self):
        self.assertTrue(
            is_strict_source_request(
                "Answer from the uploaded book and provide PDF page references."
            )
        )

    def test_pdf_tutor_is_strict(self):
        self.assertTrue(
            is_strict_source_request(
                "Explain smear layer.",
                mode="PDF Tutor",
            )
        )

    def test_general_study_question_is_not_forced_strict(self):
        self.assertFalse(
            is_strict_source_request(
                "Explain smear layer removal for my viva.",
                mode="Study Chat",
            )
        )


if __name__ == "__main__":
    unittest.main()
