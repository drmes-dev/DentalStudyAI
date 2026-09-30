"""Run the real D1 SQL against SQLite; use fake Pinecone and HTTP responses."""
import json
import sqlite3
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from d1_bank import D1BankIndex, MIGRATION_KEY
from test_engine import D1PastPaperStore, TEST_NS, TEST_MANIFEST_NS, create_past_paper_store


class SqliteIndex(D1BankIndex):
    def __init__(self, connection=None):
        super().__init__("a" * 32, "42ded302-0c9a-4f29-8e46-e523683912eb", "offline-secret")
        self.db = connection or sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self.calls = []

    def _query(self, sql, params=()):
        self.calls.append((sql, list(params)))
        self.assert_parameters(params)
        cursor = self.db.execute(sql, params)
        rows = [dict(row) for row in cursor.fetchall()] if cursor.description else []
        self.db.commit()
        return rows

    def assert_parameters(self, params):
        if len(params) > 100:
            raise AssertionError("D1 parameter limit exceeded")


class SourceStore:
    configured = True
    def __init__(self, records):
        self.records = records
        self.fetches = 0
        self.omit = False
    def connect(self):
        return self
    def _list_ids(self, namespace, prefix):
        return [key for ns, key in self.records if ns == namespace and key.startswith(prefix)]
    def fetch(self, ids, namespace):
        self.fetches += 1
        return {"vectors": {key: self.records[(namespace, key)] for key in ids if not self.omit}}


class D1BankTests(unittest.TestCase):
    def test_batches_pagination_unicode_and_idempotent_upserts(self):
        index = SqliteIndex()
        index.create_schema()
        records = [{"id": f"ppq#exam#{i:04d}", "metadata": {"stem": "مریض's question", "position": i},
                    "values": [99.0] * 1024} for i in range(1050)]
        index.upsert(records, TEST_NS)
        index.upsert(records[:2], TEST_NS)
        ids = index.list_ids(TEST_NS, "ppq#exam#")
        self.assertEqual(len(ids), 1050)
        self.assertEqual(len(index.fetch(ids, TEST_NS)["vectors"]), 1050)
        columns = [row["name"] for row in index._query("PRAGMA table_info(dentora_bank_records)")]
        self.assertNotIn("values", columns)
        index.delete(ids[:100], TEST_NS)
        self.assertEqual(len(index.list_ids(TEST_NS, "ppq#exam#")), 950)
        self.assertEqual(index.list_ids("another_namespace", "ppq#"), [])

    def test_migration_keeps_answers_and_progress_then_never_rereads_pinecone(self):
        old = {
            (TEST_NS, "ppq#exam#one"): {"metadata": {"verified_answer": "B", "verification_status": "verified", "stem": "Source?"}},
            (TEST_MANIFEST_NS, "ppmeta#exam"): {"metadata": {"processed_pages": 26, "import_complete": False}},
        }
        source = SourceStore(old)
        index = SqliteIndex()
        namespaces = [(TEST_NS, "ppq#"), (TEST_MANIFEST_NS, "ppmeta#")]
        index.ensure_ready(source, namespaces)
        self.assertEqual(index.fetch(["ppq#exam#one"], TEST_NS)["vectors"]["ppq#exam#one"]["metadata"]["verified_answer"], "B")
        self.assertEqual(index.fetch(["ppmeta#exam"], TEST_MANIFEST_NS)["vectors"]["ppmeta#exam"]["metadata"]["processed_pages"], 26)
        self.assertEqual(source.records, old)
        source.fetches = 0
        restarted = SqliteIndex(index.db)
        source.configured = False
        restarted.ensure_ready(source, namespaces)
        self.assertEqual(source.fetches, 0)

    def test_failed_migration_is_not_activated_and_can_retry(self):
        source = SourceStore({(TEST_NS, "ppq#one"): {"metadata": {"stem": "Source?"}}})
        source.omit = True
        index = SqliteIndex()
        with self.assertRaisesRegex(RuntimeError, "incomplete"):
            index.ensure_ready(source, [(TEST_NS, "ppq#")])
        self.assertFalse(index._ready)
        self.assertEqual(index._query("SELECT value FROM dentora_bank_settings WHERE key = ?", [MIGRATION_KEY]), [])
        source.omit = False
        index.ensure_ready(source, [(TEST_NS, "ppq#")])
        self.assertTrue(index._ready)

    def test_store_can_save_sample_grade_and_update_without_pinecone(self):
        index = SqliteIndex()
        index.create_schema()
        index._ready = True
        store = D1PastPaperStore()
        store._d1_index = index
        store.index_paper(paper_id="exam", title="Ortho", subject="Orthodontics", year="2025",
            filename="scan.pdf", replace_existing=False,
            questions=[{"stem": "Which tooth?", "page": 2, "options": {"A": "Incisor", "B": "Canine"}, "suggested_answer": "B"}],
            import_progress={"processed_pages": 2, "import_complete": False})
        self.assertEqual(store.catalog()["test_ready_questions"], 1)
        test = store.sample_test(count=5)
        self.assertEqual(len(test), 1)
        self.assertEqual(store.grade([{"question_id": test[0]["id"], "answer": "B"}])["score_percent"], 100)
        store.update_verification(test[0]["id"], status="verified", verified_answer="A", confidence="high", rationale="Book", sources=[])
        self.assertEqual(store.grade([{"question_id": test[0]["id"], "answer": "A"}])["score_percent"], 100)
        self.assertEqual(store.refresh_manifest_counts("exam")["verified_count"], 1)

    def test_http_quota_errors_do_not_leak_token_or_look_like_empty_bank(self):
        index = D1BankIndex("a" * 32, "42ded302-0c9a-4f29-8e46-e523683912eb", "offline-secret")
        import httpx
        with patch("d1_bank.httpx.post", return_value=httpx.Response(429)) as post:
            with self.assertRaises(RuntimeError) as error:
                index._query("SELECT 1")
        self.assertNotIn("offline-secret", str(error.exception))
        self.assertIn("429", str(error.exception))
        self.assertFalse(post.call_args.kwargs["follow_redirects"])

    def test_storage_switch_defaults_to_existing_bank(self):
        with patch.dict("os.environ", {"QUESTION_BANK_STORAGE": "pinecone"}):
            self.assertEqual(create_past_paper_store().storage_name, "pinecone")
        with patch.dict("os.environ", {"QUESTION_BANK_STORAGE": "d1"}):
            self.assertEqual(create_past_paper_store().storage_name, "d1")


if __name__ == "__main__":
    unittest.main()
