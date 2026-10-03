import unittest

from backend.rag import is_assessment_like, is_assessment_source


class EvidenceTypeTests(unittest.TestCase):
    def test_past_paper_provenance_is_not_explanatory_evidence(self):
        self.assertTrue(is_assessment_source({'category': 'Past Papers'}, 'OCR lost the option labels'))
        self.assertTrue(is_assessment_source({'filename': 'Ortho PASTPAPERS COMPILED.pdf'}, 'OCR fragment'))
        self.assertFalse(is_assessment_source({'filename': 'orthodontics.pdf', 'category': 'Textbooks'}, 'An explanatory paragraph.'))

    def test_detects_distractor_heavy_review_question(self):
        text = (
            "6. Although removal of the smear layer is not universally advocated "
            "before root canal obturation, those advocating its removal cite the "
            "following rationale. a. Smear layer provides an improved seal of the "
            "canal. b. The organic component of smear layer is antimicrobial. "
            "c. Smear layer may harbor bacteria. d. Smear layer interferes with "
            "sealer adaptation."
        )
        self.assertTrue(is_assessment_like(text))

    def test_detects_true_false_item(self):
        self.assertTrue(
            is_assessment_like(
                "NaOCl is effective in removing the smear layer. a. True b. False"
            )
        )

    def test_keeps_explanatory_prose(self):
        text = (
            "The small particles of the smear layer are primarily inorganic with "
            "a high surface-to-mass ratio, which facilitates removal by acids and "
            "chelators. EDTA is commonly used as a chelating agent."
        )
        self.assertFalse(is_assessment_like(text))

    def test_keeps_figure_caption(self):
        text = (
            "Figure 19.22. Irrigation with 17% EDTA followed by NaOCl could "
            "successfully remove the smear layer from the root canal wall."
        )
        self.assertFalse(is_assessment_like(text))

    def test_keeps_normal_lettered_expository_list(self):
        text = (
            "Clinical objectives include a. adequate access, b. preservation of "
            "tooth structure, and c. effective irrigation during preparation."
        )
        self.assertFalse(is_assessment_like(text))


if __name__ == "__main__":
    unittest.main()
