"""Store question metadata in D1 without vector payloads or public credentials."""
import json
import os
import re
from threading import RLock

import httpx


SCHEMA = """CREATE TABLE IF NOT EXISTS dentora_bank_records (
    namespace TEXT NOT NULL,
    id TEXT NOT NULL,
    metadata TEXT NOT NULL CHECK(json_valid(metadata)),
    PRIMARY KEY(namespace, id)
) WITHOUT ROWID"""
SETTINGS_SCHEMA = """CREATE TABLE IF NOT EXISTS dentora_bank_settings (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID"""
MIGRATION_KEY = "pinecone-bank-copy-v1"


def d1_configured():
    return all(os.getenv(key, "").strip() for key in (
        "CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_D1_DATABASE_ID", "CLOUDFLARE_API_TOKEN",
    ))


class D1BankIndex:
    def __init__(self, account_id=None, database_id=None, token=None):
        self.account_id = account_id or os.getenv("CLOUDFLARE_ACCOUNT_ID", "").strip()
        self.database_id = database_id or os.getenv("CLOUDFLARE_D1_DATABASE_ID", "").strip()
        self._token = token or os.getenv("CLOUDFLARE_API_TOKEN", "").strip()
        if not re.fullmatch(r"[a-fA-F0-9]{32}", self.account_id):
            raise RuntimeError("CLOUDFLARE_ACCOUNT_ID must be the 32-character account ID.")
        if not re.fullmatch(r"[a-fA-F0-9]{8}(?:-[a-fA-F0-9]{4}){3}-[a-fA-F0-9]{12}", self.database_id):
            raise RuntimeError("CLOUDFLARE_D1_DATABASE_ID must be the database UUID.")
        if not self._token:
            raise RuntimeError("CLOUDFLARE_API_TOKEN is missing.")
        self._ready = False
        self._lock = RLock()

    def _query(self, sql, params=()):
        url = ("https://api.cloudflare.com/client/v4/accounts/" + self.account_id
               + "/d1/database/" + self.database_id + "/query")
        try:
            response = httpx.post(url, headers={"Authorization": "Bearer " + self._token},
                json={"sql": sql, "params": list(params)}, timeout=30, follow_redirects=False)
        except httpx.HTTPError:
            raise RuntimeError("D1 connection failed; retry when the service is reachable.") from None
        if response.status_code != 200:
            raise RuntimeError(f"D1 request failed (HTTP {response.status_code}); check token permissions and free-plan limits.")
        try:
            body = response.json()
        except ValueError:
            raise RuntimeError("D1 returned an invalid response.") from None
        results = body.get("result")
        if not body.get("success") or not isinstance(results, list) or not results:
            raise RuntimeError("D1 query failed; check database access and free-plan limits.")
        if any(not item.get("success") for item in results):
            raise RuntimeError("D1 could not execute the bank query.")
        return results[0].get("results") or []

    def create_schema(self):
        self._query(SCHEMA)
        self._query(SETTINGS_SCHEMA)

    def list_ids(self, namespace, prefix):
        cursor = ""
        ids = []
        while True:
            rows = self._query("SELECT id FROM dentora_bank_records WHERE namespace = ? "
                "AND id >= ? AND id < ? AND id > ? ORDER BY id LIMIT 1000",
                [namespace, prefix, prefix + "\uffff", cursor])
            batch = [row["id"] for row in rows]
            ids.extend(batch)
            if len(batch) < 1000:
                return ids
            cursor = batch[-1]

    def fetch(self, ids, namespace):
        vectors = {}
        for start in range(0, len(ids), 99):
            batch = ids[start:start + 99]
            if not batch:
                continue
            rows = self._query("SELECT id, metadata FROM dentora_bank_records WHERE namespace = ? "
                "AND id IN (" + ",".join("?" for _ in batch) + ")", [namespace, *batch])
            for row in rows:
                # Compatibility value is created locally, never stored/transferred.
                vectors[row["id"]] = {"metadata": json.loads(row["metadata"]), "values": [1.0]}
        return {"vectors": vectors}

    def upsert(self, vectors, namespace):
        # Three parameters per row; keep below D1's 100-bound-parameter limit.
        for start in range(0, len(vectors), 33):
            batch = vectors[start:start + 33]
            params = []
            for vector in batch:
                params.extend([namespace, vector["id"], json.dumps(vector["metadata"],
                    ensure_ascii=False, separators=(",", ":"))])
            self._query("INSERT INTO dentora_bank_records (namespace,id,metadata) VALUES "
                + ",".join("(?,?,?)" for _ in batch)
                + " ON CONFLICT(namespace,id) DO UPDATE SET metadata=excluded.metadata", params)

    def delete(self, ids, namespace):
        for start in range(0, len(ids), 99):
            batch = ids[start:start + 99]
            self._query("DELETE FROM dentora_bank_records WHERE namespace = ? AND id IN ("
                + ",".join("?" for _ in batch) + ")", [namespace, *batch])

    def ensure_ready(self, source_store, namespaces):
        """Copy and verify old records once before allowing any D1 bank use.

        Preserve metadata verbatim, including verified answers and import offsets.
        Pinecone is never deleted. Partial copies can be retried idempotently.
        """
        with self._lock:
            if self._ready:
                return
            self.create_schema()
            marker = self._query("SELECT value FROM dentora_bank_settings WHERE key = ?", [MIGRATION_KEY])
            if marker:
                self._ready = True
                return
            if not source_store.configured:
                raise RuntimeError("Keep Pinecone configured until the existing bank has been copied to D1.")
            source = source_store.connect()
            if source is None:
                raise RuntimeError("Pinecone is unavailable; the bank migration has not completed.")
            from rag import _fetch_vectors, _metadata
            counts = {}
            for namespace, prefix in namespaces:
                ids = source_store._list_ids(namespace, prefix)
                counts[namespace] = len(ids)
                for start in range(0, len(ids), 100):
                    batch = ids[start:start + 100]
                    fetched = _fetch_vectors(source.fetch(ids=batch, namespace=namespace))
                    if set(batch) - set(fetched):
                        raise RuntimeError("The Pinecone snapshot is incomplete; retry the migration.")
                    records = [{"id": key, "metadata": _metadata(fetched[key])} for key in batch]
                    self.upsert(records, namespace)
                    copied = self.fetch(batch, namespace)["vectors"]
                    if any(key not in copied or copied[key]["metadata"] != _metadata(fetched[key]) for key in batch):
                        raise RuntimeError("D1 copy verification failed; Pinecone still contains the original bank.")
            self._query("INSERT INTO dentora_bank_settings (key,value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value", [MIGRATION_KEY, json.dumps(counts)])
            self._ready = True
