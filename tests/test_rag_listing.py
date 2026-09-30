"""Offline regression tests for Pinecone document enumeration.

These use fake Pinecone responses only; no API keys or network calls required.
Run: python -m unittest discover -s tests -v
"""

import unittest

from backend.rag import RagStore


class FakeListItem:
    def __init__(self, vector_id):
        self.id = vector_id

    def __str__(self):
        # Reproduces the bug in the old code:
        # str(ListItem) is NOT a valid vector identifier.
        return "ListItem(id=" + repr(self.id) + ")"


class FakeListResponse:
    def __init__(self, vector_ids):
        self.vectors = [FakeListItem(item) for item in vector_ids]

    def __iter__(self):
        return iter(self.vectors)


class FakeIndex:
    def __init__(self, pages=None):
        self.pages = pages or []
        self.requested = []
        self.fetched = []

    def list(self, namespace, prefix):
        self.requested.append((namespace, prefix))
        return iter(self.pages)

    def fetch(self, ids, namespace):
        self.fetched.append((list(ids), namespace))
        vectors = {}
        for vector_id in ids:
            if vector_id.startswith("docmeta#"):
                vectors[vector_id] = {
                    "metadata": {
                        "record_type": "document",
                        "doc_id": vector_id[len("docmeta#"):],
                        "filename": "example.pdf",
                        "title": "Example",
                        "category": "Books",
                        "indexed_at": "2026-09-30T00:00:00Z",
                        "pages": 100,
                        "chunks": 150,
                    }
                }
        return {"vectors": vectors}


def make_store(fake_index):
    store = RagStore()
    store.connect = lambda: fake_index
    return store


class PineconeListingTests(unittest.TestCase):
    def test_sdk_listresponse_objects_are_read_as_ids(self):
        index = FakeIndex([
            FakeListResponse(["docmeta#abc123", "docmeta#def456"]),
        ])
        store = make_store(index)

        self.assertEqual(
            store._list_ids("manifest", "docmeta#"),
            ["docmeta#abc123", "docmeta#def456"],
        )
        self.assertEqual(index.requested, [("manifest", "docmeta#")])

    def test_older_list_and_dictionary_pages_still_work(self):
        index = FakeIndex([
            ["docmeta#abc123"],
            {"vectors": [{"id": "docmeta#def456"}]},
            FakeListResponse(["docmeta#abc123", "unrelated#skip"]),
        ])
        store = make_store(index)

        self.assertEqual(
            store._list_ids("manifest", "docmeta#"),
            ["docmeta#abc123", "docmeta#def456"],
        )

    def test_library_fetches_two_real_manifest_ids(self):
        index = FakeIndex([
            FakeListResponse(["docmeta#book1", "docmeta#book2"]),
        ])
        store = make_store(index)

        documents = store.list_documents()

        self.assertEqual(len(documents), 2)
        self.assertEqual(
            {doc["doc_id"] for doc in documents},
            {"book1", "book2"},
        )
        self.assertEqual(
            index.fetched,
            [(["docmeta#book1", "docmeta#book2"], "manifest")],
        )

    def test_listing_error_must_not_silently_look_like_empty_library(self):
        class ErrorIndex:
            def list(self, namespace, prefix):
                raise RuntimeError("remote index unavailable")

        store = make_store(ErrorIndex())

        with self.assertRaisesRegex(RuntimeError, "Could not list Pinecone"):
            store._list_ids("manifest", "docmeta#")


if __name__ == "__main__":
    unittest.main()
