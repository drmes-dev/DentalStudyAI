import hashlib
import json
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional

from rag import EMBED_DIMENSION, _fetch_vectors, _metadata, now_iso, normalize_text, rag_store


TEST_NS = os.getenv("PINECONE_TEST_NAMESPACE", "past_papers").strip()
TEST_MANIFEST_NS = os.getenv(
    "PINECONE_TEST_MANIFEST_NAMESPACE",
    "past_papers_manifest",
).strip()


def _clean(value: Any, limit: int = 500) -> str:
    return normalize_text(str(value or ""))[:limit]


def _json(value: Any, limit: int = 12000) -> str:
    raw = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return raw[:limit]


def _loads(value: Any, fallback: Any):
    if not value:
        return fallback
    try:
        return json.loads(str(value))
    except Exception:
        return fallback


def _stem_hash(stem: str) -> str:
    compact = " ".join(
        token
        for token in "".join(
            char.lower() if char.isalnum() else " "
            for char in str(stem or "")
        ).split()
        if token
    )
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()[:20]


def _question_text(question: Dict[str, Any]) -> str:
    options = question.get("options") or {}
    option_text = " ".join(
        f"{key}. {value}"
        for key, value in options.items()
    )
    return normalize_text(
        f"{question.get('stem', '')} {option_text} "
        f"Topic: {question.get('topic', '')}. "
        f"Subtopic: {question.get('subtopic', '')}."
    )


class PastPaperStore:
    @property
    def configured(self) -> bool:
        return rag_store.configured

    def _index(self):
        index = rag_store.connect()
        if index is None:
            raise RuntimeError("Persistent Pinecone storage is not configured.")
        return index

    def _manifest_id(self, paper_id: str) -> str:
        return f"ppmeta#{paper_id}"

    def _question_vector_id(self, paper_id: str, question_id: str) -> str:
        return f"ppq#{paper_id}#{question_id}"

    def _delete_prefix(self, namespace: str, prefix: str):
        index = self._index()
        ids = rag_store._list_ids(namespace, prefix)
        for start in range(0, len(ids), 1000):
            batch = ids[start:start + 1000]
            if batch:
                index.delete(ids=batch, namespace=namespace)
        return len(ids)

    def index_paper(
        self,
        *,
        paper_id: str,
        title: str,
        subject: str,
        year: str,
        filename: str,
        questions: List[Dict[str, Any]],
        replace_existing: bool = True,
    ) -> Dict[str, Any]:
        index = self._index()

        if replace_existing:
            self._delete_prefix(TEST_NS, f"ppq#{paper_id}#")

        clean_questions = []
        for position, raw in enumerate(questions, start=1):
            stem = _clean(raw.get("stem"), 4000)
            if not stem:
                continue

            options = raw.get("options") or {}
            if isinstance(options, list):
                options = {
                    chr(65 + idx): _clean(value, 1200)
                    for idx, value in enumerate(options[:8])
                }
            elif isinstance(options, dict):
                options = {
                    str(key).strip().upper()[:4]: _clean(value, 1200)
                    for key, value in options.items()
                    if _clean(value, 1200)
                }
            else:
                options = {}

            question_id = _clean(raw.get("question_id"), 80)
            if not question_id:
                question_id = hashlib.sha256(
                    f"{paper_id}|{position}|{stem}".encode("utf-8")
                ).hexdigest()[:16]

            item = {
                "question_id": question_id,
                "question_number": _clean(raw.get("question_number") or position, 80),
                "stem": stem,
                "options": options,
                "question_type": _clean(raw.get("question_type") or ("mcq" if options else "short_answer"), 40).lower(),
                "topic": _clean(raw.get("topic") or "Unclassified", 160),
                "subtopic": _clean(raw.get("subtopic") or "", 180),
                "page": int(raw.get("page") or 0),
                "marks": float(raw.get("marks") or 0),
                "provided_answer": _clean(raw.get("provided_answer"), 40).upper(),
                "verification_status": "pending" if options else "not_auto_gradable",
                "verified_answer": "",
                "verification_confidence": "",
                "verification_rationale": "",
                "verification_sources": [],
                "stem_hash": _stem_hash(stem),
            }
            clean_questions.append(item)

        if not clean_questions:
            raise ValueError("No usable questions were parsed from this paper.")

        vectors = rag_store.embed(
            [_question_text(item) for item in clean_questions],
            "passage",
        )

        indexed_at = now_iso()
        records = []

        for item, vector in zip(clean_questions, vectors):
            metadata = {
                "record_type": "past_paper_question",
                "paper_id": paper_id,
                "paper_title": _clean(title, 300),
                "subject": _clean(subject or "Unspecified", 160),
                "year": _clean(year or "Unknown", 40),
                "filename": _clean(filename, 300),
                "question_id": item["question_id"],
                "question_number": item["question_number"],
                "stem": item["stem"],
                "options_json": _json(item["options"]),
                "question_type": item["question_type"],
                "topic": item["topic"],
                "subtopic": item["subtopic"],
                "page": item["page"],
                "marks": item["marks"],
                "provided_answer": item["provided_answer"],
                "verification_status": item["verification_status"],
                "verified_answer": "",
                "verification_confidence": "",
                "verification_rationale": "",
                "verification_sources_json": "[]",
                "stem_hash": item["stem_hash"],
                "indexed_at": indexed_at,
            }

            records.append({
                "id": self._question_vector_id(
                    paper_id,
                    item["question_id"],
                ),
                "values": vector,
                "metadata": metadata,
            })

        for start in range(0, len(records), 50):
            index.upsert(
                vectors=records[start:start + 50],
                namespace=TEST_NS,
            )

        manifest_vector = rag_store.embed(
            [
                f"{title}. {subject}. {year}. "
                f"Past paper with {len(records)} structured questions."
            ],
            "passage",
        )[0]

        index.upsert(
            vectors=[{
                "id": self._manifest_id(paper_id),
                "values": manifest_vector,
                "metadata": {
                    "record_type": "past_paper",
                    "paper_id": paper_id,
                    "title": _clean(title, 300),
                    "subject": _clean(subject or "Unspecified", 160),
                    "year": _clean(year or "Unknown", 40),
                    "filename": _clean(filename, 300),
                    "question_count": len(records),
                    "verified_count": 0,
                    "conflict_count": 0,
                    "insufficient_count": 0,
                    "pending_count": sum(
                        1
                        for item in clean_questions
                        if item["verification_status"] == "pending"
                    ),
                    "indexed_at": indexed_at,
                },
            }],
            namespace=TEST_MANIFEST_NS,
        )

        return {
            "success": True,
            "paper_id": paper_id,
            "title": title,
            "subject": subject,
            "year": year,
            "question_count": len(records),
            "pending_count": sum(
                1
                for item in clean_questions
                if item["verification_status"] == "pending"
            ),
        }

    def _metadata_to_question(
        self,
        vector_id: str,
        metadata: Dict[str, Any],
    ) -> Dict[str, Any]:
        return {
            "id": vector_id,
            "paper_id": metadata.get("paper_id", ""),
            "paper_title": metadata.get("paper_title", ""),
            "subject": metadata.get("subject", ""),
            "year": metadata.get("year", ""),
            "filename": metadata.get("filename", ""),
            "question_id": metadata.get("question_id", ""),
            "question_number": metadata.get("question_number", ""),
            "stem": metadata.get("stem", ""),
            "options": _loads(metadata.get("options_json"), {}),
            "question_type": metadata.get("question_type", ""),
            "topic": metadata.get("topic", "Unclassified"),
            "subtopic": metadata.get("subtopic", ""),
            "page": int(metadata.get("page", 0) or 0),
            "marks": float(metadata.get("marks", 0) or 0),
            "provided_answer": metadata.get("provided_answer", ""),
            "verification_status": metadata.get("verification_status", ""),
            "verified_answer": metadata.get("verified_answer", ""),
            "verification_confidence": metadata.get("verification_confidence", ""),
            "verification_rationale": metadata.get("verification_rationale", ""),
            "verification_sources": _loads(
                metadata.get("verification_sources_json"),
                [],
            ),
            "stem_hash": metadata.get("stem_hash", ""),
        }

    def list_questions(
        self,
        *,
        paper_id: Optional[str] = None,
        subject: Optional[str] = None,
        year: Optional[str] = None,
        topic: Optional[str] = None,
        statuses: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        index = self._index()
        prefix = f"ppq#{paper_id}#" if paper_id else "ppq#"
        ids = rag_store._list_ids(TEST_NS, prefix)
        result = []

        for start in range(0, len(ids), 100):
            batch = ids[start:start + 100]
            if not batch:
                continue

            fetched = index.fetch(
                ids=batch,
                namespace=TEST_NS,
            )

            for vector_id, vector in _fetch_vectors(fetched).items():
                metadata = _metadata(vector)
                if metadata.get("record_type") != "past_paper_question":
                    continue

                item = self._metadata_to_question(
                    vector_id,
                    metadata,
                )

                if subject and item["subject"].lower() != subject.lower():
                    continue
                if year and str(item["year"]) != str(year):
                    continue
                if topic and item["topic"].lower() != topic.lower():
                    continue
                if statuses and item["verification_status"] not in statuses:
                    continue

                result.append(item)

        result.sort(
            key=lambda item: (
                item["subject"],
                item["year"],
                item["paper_title"],
                str(item["question_number"]),
            )
        )
        return result

    def fetch_questions(
        self,
        ids: List[str],
    ) -> List[Dict[str, Any]]:
        if not ids:
            return []

        index = self._index()
        result = []

        for start in range(0, len(ids), 100):
            batch = ids[start:start + 100]
            fetched = index.fetch(
                ids=batch,
                namespace=TEST_NS,
            )
            vectors = _fetch_vectors(fetched)
            for vector_id in batch:
                vector = vectors.get(vector_id)
                if vector is None:
                    continue
                metadata = _metadata(vector)
                if metadata.get("record_type") == "past_paper_question":
                    result.append(
                        self._metadata_to_question(
                            vector_id,
                            metadata,
                        )
                    )

        return result

    def pending_questions(
        self,
        paper_id: str,
        limit: int = 8,
    ) -> List[Dict[str, Any]]:
        questions = self.list_questions(
            paper_id=paper_id,
            statuses=["pending"],
        )
        return questions[:max(1, min(int(limit), 12))]

    def update_verification(
        self,
        question_id: str,
        *,
        status: str,
        verified_answer: str,
        confidence: str,
        rationale: str,
        sources: List[Dict[str, Any]],
    ):
        index = self._index()
        fetched = index.fetch(
            ids=[question_id],
            namespace=TEST_NS,
        )
        vector = _fetch_vectors(fetched).get(question_id)

        if vector is None:
            raise KeyError("Question not found.")

        metadata = _metadata(vector)
        metadata["verification_status"] = _clean(status, 40).lower()
        metadata["verified_answer"] = _clean(
            verified_answer,
            40,
        ).upper()
        metadata["verification_confidence"] = _clean(
            confidence,
            40,
        ).lower()
        metadata["verification_rationale"] = _clean(
            rationale,
            1800,
        )
        metadata["verification_sources_json"] = _json(
            sources,
            8000,
        )

        values = (
            vector.get("values")
            if isinstance(vector, dict)
            else getattr(vector, "values", None)
        )

        if values is None:
            values = rag_store.embed(
                [_question_text(metadata)],
                "passage",
            )[0]

        index.upsert(
            vectors=[{
                "id": question_id,
                "values": list(values),
                "metadata": metadata,
            }],
            namespace=TEST_NS,
        )

    def refresh_manifest_counts(self, paper_id: str) -> Dict[str, int]:
        index = self._index()
        questions = self.list_questions(paper_id=paper_id)
        counts = Counter(
            item["verification_status"]
            for item in questions
        )

        manifest_id = self._manifest_id(paper_id)
        fetched = index.fetch(
            ids=[manifest_id],
            namespace=TEST_MANIFEST_NS,
        )
        vector = _fetch_vectors(fetched).get(manifest_id)

        if vector is not None:
            metadata = _metadata(vector)
            metadata["question_count"] = len(questions)
            metadata["verified_count"] = counts.get("verified", 0)
            metadata["conflict_count"] = counts.get("conflict", 0)
            metadata["insufficient_count"] = counts.get("insufficient", 0)
            metadata["pending_count"] = counts.get("pending", 0)

            values = (
                vector.get("values")
                if isinstance(vector, dict)
                else getattr(vector, "values", None)
            )
            if values is None:
                values = [0.0] * EMBED_DIMENSION

            index.upsert(
                vectors=[{
                    "id": manifest_id,
                    "values": list(values),
                    "metadata": metadata,
                }],
                namespace=TEST_MANIFEST_NS,
            )

        return {
            "question_count": len(questions),
            "verified_count": counts.get("verified", 0),
            "conflict_count": counts.get("conflict", 0),
            "insufficient_count": counts.get("insufficient", 0),
            "pending_count": counts.get("pending", 0),
        }

    def list_papers(self) -> List[Dict[str, Any]]:
        index = self._index()
        ids = rag_store._list_ids(
            TEST_MANIFEST_NS,
            "ppmeta#",
        )
        papers = []

        for start in range(0, len(ids), 100):
            batch = ids[start:start + 100]
            fetched = index.fetch(
                ids=batch,
                namespace=TEST_MANIFEST_NS,
            )

            for vector in _fetch_vectors(fetched).values():
                metadata = _metadata(vector)
                if metadata.get("record_type") == "past_paper":
                    papers.append(dict(metadata))

        papers.sort(
            key=lambda item: (
                str(item.get("subject", "")),
                str(item.get("year", "")),
                str(item.get("title", "")),
            )
        )
        return papers

    def catalog(self) -> Dict[str, Any]:
        questions = self.list_questions()
        papers = self.list_papers()

        repeat_counts = Counter(
            item["stem_hash"]
            for item in questions
            if item.get("stem_hash")
        )

        topics = Counter()
        subjects = Counter()
        years = Counter()
        eligible = 0

        for item in questions:
            topics[
                f"{item['subject']}|||{item['topic']}"
            ] += 1
            subjects[item["subject"]] += 1
            years[str(item["year"])] += 1

            if item["verification_status"] in {
                "verified",
                "conflict",
            } and item.get("verified_answer"):
                eligible += 1

        return {
            "configured": True,
            "paper_count": len(papers),
            "question_count": len(questions),
            "eligible_test_questions": eligible,
            "subjects": [
                {"name": key, "count": value}
                for key, value in sorted(subjects.items())
            ],
            "years": [
                {"name": key, "count": value}
                for key, value in sorted(years.items())
            ],
            "topics": [
                {
                    "subject": key.split("|||", 1)[0],
                    "name": key.split("|||", 1)[1],
                    "count": value,
                }
                for key, value in sorted(topics.items())
            ],
            "papers": papers,
            "repeated_question_groups": sum(
                1
                for value in repeat_counts.values()
                if value > 1
            ),
        }

    def sample_test(
        self,
        *,
        count: int,
        subject: Optional[str] = None,
        year: Optional[str] = None,
        topic: Optional[str] = None,
        weak_topics: Optional[List[str]] = None,
        repeated_only: bool = False,
    ) -> List[Dict[str, Any]]:
        questions = self.list_questions(
            subject=subject,
            year=year,
            topic=topic,
            statuses=["verified", "conflict"],
        )

        questions = [
            item
            for item in questions
            if item.get("verified_answer")
            and len(item.get("options") or {}) >= 2
        ]

        repeat_counts = Counter(
            item["stem_hash"]
            for item in self.list_questions()
            if item.get("stem_hash")
        )

        if repeated_only:
            questions = [
                item
                for item in questions
                if repeat_counts.get(item["stem_hash"], 0) > 1
            ]

        weak = {
            str(value).strip().lower()
            for value in (weak_topics or [])
            if str(value).strip()
        }

        if weak:
            priority = [
                item
                for item in questions
                if item["topic"].lower() in weak
            ]
            rest = [
                item
                for item in questions
                if item["topic"].lower() not in weak
            ]
            random.SystemRandom().shuffle(priority)
            random.SystemRandom().shuffle(rest)
            questions = priority + rest
        else:
            random.SystemRandom().shuffle(questions)

        requested = max(1, min(int(count), 100))
        selected = []
        used_stems = set()

        for item in questions:
            stem_key = item.get("stem_hash") or item["id"]
            if stem_key in used_stems:
                continue

            used_stems.add(stem_key)
            selected.append(item)

            if len(selected) >= requested:
                break

        return [
            {
                "id": item["id"],
                "paper_id": item["paper_id"],
                "paper_title": item["paper_title"],
                "subject": item["subject"],
                "year": item["year"],
                "question_number": item["question_number"],
                "stem": item["stem"],
                "options": item["options"],
                "topic": item["topic"],
                "subtopic": item["subtopic"],
                "page": item["page"],
                "repeat_count": repeat_counts.get(
                    item["stem_hash"],
                    1,
                ),
            }
            for item in selected
        ]

    def grade(
        self,
        responses: List[Dict[str, str]],
    ) -> Dict[str, Any]:
        by_id = {
            str(item.get("question_id", "")): str(
                item.get("answer", "")
            ).strip().upper()
            for item in responses
            if item.get("question_id")
        }

        questions = self.fetch_questions(list(by_id.keys()))
        repeat_counts = Counter(
            item["stem_hash"]
            for item in self.list_questions()
            if item.get("stem_hash")
        )

        details = []
        correct = 0
        answered = 0

        for item in questions:
            chosen = by_id.get(item["id"], "")
            expected = str(item.get("verified_answer", "")).upper()
            is_answered = bool(chosen)
            is_correct = bool(
                is_answered
                and expected
                and chosen == expected
            )

            if is_answered:
                answered += 1
            if is_correct:
                correct += 1

            details.append({
                "question_id": item["id"],
                "stem": item["stem"],
                "chosen_answer": chosen,
                "correct_answer": expected,
                "correct": is_correct,
                "answered": is_answered,
                "subject": item["subject"],
                "topic": item["topic"],
                "subtopic": item["subtopic"],
                "paper_title": item["paper_title"],
                "year": item["year"],
                "repeat_count": repeat_counts.get(
                    item["stem_hash"],
                    1,
                ),
                "verification_status": item["verification_status"],
                "verification_confidence": item["verification_confidence"],
                "verification_rationale": item["verification_rationale"],
                "verification_sources": item["verification_sources"],
            })

        total = len(details)
        return {
            "total": total,
            "answered": answered,
            "unanswered": max(0, total - answered),
            "correct": correct,
            "incorrect": max(0, answered - correct),
            "score_percent": round(
                (correct / total * 100.0)
                if total
                else 0.0,
                1,
            ),
            "details": details,
        }


past_paper_store = PastPaperStore()
