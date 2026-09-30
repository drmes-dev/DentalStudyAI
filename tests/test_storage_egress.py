"""Prevent repeated full-document downloads and stale bank cache reads."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from rag import RagStore, KNOWLEDGE_NS
from read_cache import ReadCache
from test_engine import PastPaperStore, TEST_NS


class StorageEgressTests(unittest.TestCase):
    def test_549_page_import_downloads_source_once_across_275_batches(self):
        class Index:
            def __init__(self):
                self.fetches = 0
                self.deleted = False

            def list(self, namespace, prefix):
                ids = [] if self.deleted else [f"doc#exam#p{i:04d}#c{i:05d}" for i in range(1, 550)]
                return iter([ids])

            def fetch(self, ids, namespace):
                self.fetches += 1
                return {"vectors": {key: {"metadata": {
                    "record_type": "chunk", "page": int(key.split("#p")[1][:4]),
                    "chunk_index": 0, "text": "Source exam question text",
                }} for key in ids}}

            def delete(self, ids, namespace):
                if namespace == KNOWLEDGE_NS:
                    self.deleted = True

        index = Index()
        store = RagStore()
        store.connect = lambda: index
        for batch in range(275):
            pages = store.document_pages("exam")
            self.assertEqual(len(pages), 549)
            pages[0]["text"] = "Caller must not change cached source"
        self.assertEqual(index.fetches, 6)
        self.assertEqual(store.document_pages("exam")[0]["text"], "Source exam question text")
        store.delete_document("exam", require_present=False)
        self.assertEqual(store.document_pages("exam"), [])

    def test_bank_cache_reuses_reads_and_invalidates_after_append_and_verification(self):
        import test_engine
        records = {}
        class Index:
            def __init__(self):
                self.fetches = 0
            def upsert(self, vectors, namespace):
                for vector in vectors:
                    records[(namespace, vector["id"])] = vector
            def fetch(self, ids, namespace):
                self.fetches += 1
                return {"vectors": {key: records[(namespace, key)] for key in ids if (namespace, key) in records}}
        index = Index()
        store = PastPaperStore()
        store._index = lambda: index
        def ids(namespace, prefix):
            return [key for ns, key in records if ns == namespace and key.startswith(prefix)]
        def append(stem):
            return store.index_paper(paper_id="exam", title="Exam", subject="Ortho", year="2025",
                filename="exam.pdf", replace_existing=False,
                questions=[{"stem": stem, "options": {"A": "One", "B": "Two"}, "suggested_answer": "B"}])
        with patch.object(test_engine.rag_store, "_list_ids", side_effect=ids):
            append("First source question")
            first = store.catalog()
            fetches = index.fetches
            for _ in range(10):
                self.assertEqual(store.catalog()["mcq_count"], 1)
            self.assertEqual(index.fetches, fetches)
            append("Second source question")
            self.assertEqual(store.catalog()["mcq_count"], 2)
            question = store.list_questions()[0]
            store.update_verification(question["id"], status="verified", verified_answer="A",
                confidence="high", rationale="Source", sources=[])
            updated = next(q for q in store.list_questions() if q["id"] == question["id"])
            self.assertEqual(updated["verified_answer"], "A")

    def test_cache_is_bounded_and_does_not_save_failed_reads(self):
        cache = ReadCache(max_bytes=20)
        loads = []
        def read(key):
            return cache.read(key, lambda: loads.append(key) or [key], ttl=None)
        read("a" * 10)
        read("b" * 10)
        read("a" * 10)
        self.assertEqual(len(loads), 3)
        self.assertLessEqual(cache.bytes, 20)
        with self.assertRaises(RuntimeError):
            cache.read("error", lambda: (_ for _ in ()).throw(RuntimeError("Remote failure")), ttl=None)
        self.assertEqual(cache.read("error", lambda: ["ok"], ttl=None), ["ok"])


if __name__ == "__main__":
    unittest.main()
