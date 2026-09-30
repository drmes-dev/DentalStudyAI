"""Small, explicit exports for on-device practice; no embeddings or PDFs."""
import json
from collections import Counter
from datetime import datetime, timezone


def build_offline_pack(questions, papers):
    repeats = Counter(q.get("stem_hash") or q["id"] for q in questions)
    records = []
    for question in questions:
        options = question.get("options") or {}
        if len(options) < 2:
            continue
        reviewed = str(question.get("verified_answer") or "").upper()
        printed = str(question.get("provided_answer") or "").upper()
        provisional = str(question.get("provisional_answer") or "").upper()
        expected = reviewed or printed or provisional
        if expected not in options:
            continue
        item = {key: question.get(key, "") for key in (
            "id", "paper_id", "paper_title", "subject", "year", "question_number",
            "stem", "options", "topic", "subtopic", "page", "verification_status",
            "verification_confidence", "verification_rationale", "verification_sources")}
        item.update(
            correct_answer=expected,
            answer_source="textbook_review" if reviewed else "past_paper_key" if printed else "ai_prepared",
            trusted_answer=bool((reviewed and question.get("verification_status") == "verified")
                                or (not reviewed and printed and question.get("verification_status") != "conflict")),
            stem_hash=question.get("stem_hash") or question["id"],
            repeat_count=repeats[question.get("stem_hash") or question["id"]],
        )
        records.append(item)
    included = {q["paper_id"] for q in records}
    pack = {"schema_version": 1, "saved_at": datetime.now(timezone.utc).isoformat(),
            "questions": records, "papers": [{key: p.get(key, "") for key in (
                "paper_id", "title", "subject", "year")} for p in papers if p.get("paper_id") in included]}
    if len(json.dumps(pack, ensure_ascii=False).encode("utf-8")) > 8 * 1024 * 1024:
        raise ValueError("This pack exceeds 8 MB. Choose a subject before saving offline.")
    return pack
