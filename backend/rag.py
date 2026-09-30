import hashlib
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from pinecone import Pinecone, ServerlessSpec
except Exception:
    Pinecone = None
    ServerlessSpec = None


PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "").strip()
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "dentora-knowledge").strip()
CLOUD = os.getenv("PINECONE_CLOUD", "aws").strip()
REGION = os.getenv("PINECONE_REGION", "us-east-1").strip()
EMBED_MODEL = os.getenv("PINECONE_EMBED_MODEL", "llama-text-embed-v2").strip()
EMBED_DIMENSION = int(os.getenv("PINECONE_EMBED_DIMENSION", "1024"))

KNOWLEDGE_NS = os.getenv("PINECONE_RAG_NAMESPACE", "knowledge").strip()
MANIFEST_NS = os.getenv("PINECONE_MANIFEST_NAMESPACE", "manifest").strip()

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "1800"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "250"))
RAG_CANDIDATES = int(os.getenv("RAG_CANDIDATES", "18"))
RAG_CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "7"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.20"))

VALID_CATEGORIES = {
    "Books",
    "Past Papers",
    "Viva",
    "OSCE",
    "Notes",
    "Other",
}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(text: str) -> str:
    text = str(text or "").replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_category(category: str) -> str:
    category = str(category or "Other").strip()
    return category if category in VALID_CATEGORIES else "Other"


def document_id(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:24]


def chunk_page_text(text: str, page: int) -> List[Dict[str, Any]]:
    text = normalize_text(text)
    if not text:
        return []

    result = []
    start = 0
    local_index = 0

    while start < len(text):
        raw_end = min(start + CHUNK_SIZE, len(text))
        end = raw_end

        if raw_end < len(text):
            boundary = text.rfind(" ", start + int(CHUNK_SIZE * 0.65), raw_end)
            if boundary > start:
                end = boundary

        value = text[start:end].strip()
        if value:
            result.append({
                "page": int(page),
                "local_chunk_index": local_index,
                "text": value,
            })
            local_index += 1

        if end >= len(text):
            break

        start = max(end - CHUNK_OVERLAP, start + 1)

    return result


def chunk_pages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    global_index = 0

    for page_data in pages:
        for chunk in chunk_page_text(
            page_data.get("text", ""),
            int(page_data.get("page", 0)),
        ):
            chunk["chunk_index"] = global_index
            global_index += 1
            result.append(chunk)

    return result


def keyword_overlap(query: str, text: str, title: str = "") -> float:
    words = {
        word
        for word in re.findall(r"[a-zA-Z0-9]+", str(query).lower())
        if len(word) > 2
    }
    if not words:
        return 0.0

    haystack = f"{title} {text}".lower()
    hits = sum(1 for word in words if word in haystack)
    return hits / len(words)


def is_assessment_like(text: str) -> bool:
    """Detect chunks that are likely review questions/MCQ answer options.

    This is intentionally conservative: normal explanatory prose should remain
    eligible, while obvious question banks and distractor-heavy chunks are
    kept out of evidence grounding in non-MCQ modes.
    """
    value = normalize_text(text)
    if not value:
        return False

    lower = value.lower()
    compact = re.sub(r"\s+", " ", lower)
    score = 0

    strong_cues = (
        "review question",
        "review questions",
        "self-assessment",
        "self assessment",
        "multiple-choice",
        "multiple choice",
        "true or false",
    )
    if any(cue in compact for cue in strong_cues):
        score += 5

    if re.search(r"\ba[\.)]\s*true\b.{0,80}\bb[\.)]\s*false\b", compact):
        score += 5

    option_count = len(re.findall(r"(?:^|\s)[a-e][\.)]\s+", lower))
    if option_count >= 3:
        score += 3
    elif option_count == 2:
        score += 1

    question_cues = (
        "which of the following",
        "all of the following",
        "choose the best",
        "choose the correct",
        "select the best",
        "select the correct",
        "is correct except",
        "is true except",
        "is false except",
    )
    if any(cue in compact for cue in question_cues):
        score += 3

    if option_count >= 2 and re.match(r"^\s*\d{1,3}[\.)]\s+", value):
        score += 2

    if option_count >= 2 and value.count("?") >= 1:
        score += 2

    return score >= 4

def is_strict_source_request(message: str, mode: str = "") -> bool:
    """Return True when the student explicitly wants a source-bound answer."""
    text = re.sub(r"\s+", " ", str(message or "").lower()).strip()
    mode_key = str(mode or "").strip().lower()

    if mode_key in {"pdf tutor", "pdf", "source tutor"}:
        return True

    cues = (
        "according to the uploaded",
        "according to uploaded",
        "according to the textbook",
        "according to this textbook",
        "uploaded textbook",
        "uploaded book",
        "uploaded resource",
        "uploaded resources",
        "provided textbook",
        "provided book",
        "from the uploaded",
        "from this book",
        "from this textbook",
        "textbook page reference",
        "textbook page references",
        "pdf page reference",
        "pdf page references",
        "do not use information from",
        "do not use outside",
        "only use the uploaded",
        "use only the uploaded",
    )
    return any(cue in text for cue in cues)


def _matches(result) -> List[Any]:
    if isinstance(result, dict):
        return list(result.get("matches", []) or [])
    return list(getattr(result, "matches", []) or [])


def _score(match) -> float:
    if isinstance(match, dict):
        return float(match.get("score", 0.0) or 0.0)
    return float(getattr(match, "score", 0.0) or 0.0)


def _metadata(item) -> Dict[str, Any]:
    if isinstance(item, dict):
        return dict(item.get("metadata", {}) or {})
    return dict(getattr(item, "metadata", {}) or {})


def _fetch_vectors(result) -> Dict[str, Any]:
    if isinstance(result, dict):
        return dict(result.get("vectors", {}) or {})
    return dict(getattr(result, "vectors", {}) or {})


def _embedding_values(item) -> List[float]:
    if isinstance(item, dict):
        values = item.get("values")
    else:
        values = getattr(item, "values", None)
    if values is None:
        raise RuntimeError("Pinecone inference returned no embedding values.")
    return list(values)


class RagStore:
    def __init__(self):
        self.pc = None
        self.index = None

    @property
    def configured(self) -> bool:
        return bool(PINECONE_API_KEY and Pinecone is not None and ServerlessSpec is not None)

    def _index_names(self) -> List[str]:
        result = self.pc.list_indexes()
        if hasattr(result, "names"):
            return list(result.names())

        names = []
        try:
            for item in result:
                name = item.get("name") if isinstance(item, dict) else getattr(item, "name", None)
                if name:
                    names.append(name)
        except Exception:
            pass
        return names

    def connect(self):
        if not self.configured:
            return None

        if self.index is not None:
            return self.index

        self.pc = Pinecone(api_key=PINECONE_API_KEY)

        if INDEX_NAME not in self._index_names():
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=EMBED_DIMENSION,
                metric="cosine",
                spec=ServerlessSpec(cloud=CLOUD, region=REGION),
            )

            for _ in range(30):
                try:
                    description = self.pc.describe_index(INDEX_NAME)
                    status = description.get("status") if isinstance(description, dict) else getattr(description, "status", None)
                    ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
                    if ready:
                        break
                except Exception:
                    pass
                time.sleep(1)

        self.index = self.pc.Index(INDEX_NAME)
        return self.index

    def embed(self, texts: List[str], input_type: str) -> List[List[float]]:
        if not texts:
            return []

        self.connect()
        if self.pc is None:
            raise RuntimeError("Persistent RAG is not configured.")

        vectors: List[List[float]] = []

        for start in range(0, len(texts), 32):
            batch = texts[start:start + 32]
            last_error = None

            for attempt in range(4):
                try:
                    response = self.pc.inference.embed(
                        model=EMBED_MODEL,
                        inputs=batch,
                        parameters={
                            "input_type": input_type,
                            "truncate": "END",
                        },
                    )
                    vectors.extend(_embedding_values(item) for item in response)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        time.sleep(2 ** attempt)

            if last_error is not None:
                raise last_error

        return vectors

    def manifest_id(self, doc_id: str) -> str:
        return f"docmeta#{doc_id}"

    def manifest_exists(self, doc_id: str) -> bool:
        index = self.connect()
        if index is None:
            return False

        result = index.fetch(
            ids=[self.manifest_id(doc_id)],
            namespace=MANIFEST_NS,
        )
        return bool(_fetch_vectors(result))

    def _list_ids(self, namespace: str, prefix: str) -> List[str]:
        """Return actual vector IDs across Pinecone SDK response versions.

        Recent Pinecone SDKs yield a ListResponse with a .vectors collection
        of ListItem objects. Calling str(ListItem) produces a representation,
        not its ID, and silently breaks subsequent index.fetch() calls.
        """
        index = self.connect()
        if index is None:
            return []

        def get_id(item: Any) -> Optional[str]:
            if isinstance(item, str):
                return item
            if isinstance(item, dict):
                return item.get("id") or item.get("_id")
            return getattr(item, "id", None) or getattr(item, "_id", None)

        ids: List[str] = []
        try:
            for page in index.list(namespace=namespace, prefix=prefix):
                if isinstance(page, (list, tuple)):
                    entries = page
                elif isinstance(page, dict):
                    entries = page.get("vectors") or page.get("ids") or []
                else:
                    entries = getattr(page, "vectors", None)
                    if entries is None:
                        entries = getattr(page, "ids", None)
                    if entries is None:
                        entries = list(page)

                for item in entries:
                    vector_id = get_id(item)
                    if vector_id and str(vector_id).startswith(prefix):
                        ids.append(str(vector_id))
        except Exception as exc:
            raise RuntimeError(
                f"Could not list Pinecone records in namespace {namespace!r}: {exc}"
            ) from exc

        # Preserve order and avoid duplicate IDs across SDK pagination.
        return list(dict.fromkeys(ids))

    def index_pages(
        self,
        *,
        doc_id: str,
        filename: str,
        title: str,
        category: str,
        pages: List[Dict[str, Any]],
        source_url: str = "",
        replace_existing: bool = False,
    ) -> Dict[str, Any]:
        index = self.connect()
        if index is None:
            raise RuntimeError("Persistent RAG is not configured.")

        category = safe_category(category)
        chunks = chunk_pages(pages)

        if self.manifest_exists(doc_id) and not replace_existing:
            return {
                "success": True,
                "duplicate": True,
                "doc_id": doc_id,
                "filename": filename,
                "title": title,
                "category": category,
                "pages": len(pages),
                "chunks": len(chunks),
                "message": "This exact document is already indexed.",
            }

        if replace_existing:
            self.delete_document(doc_id, require_present=False)

        if not chunks:
            raise RuntimeError("No readable text was extracted from this document.")

        vectors = self.embed([chunk["text"] for chunk in chunks], "passage")
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding count did not match chunk count.")

        indexed_at = now_iso()
        records = []

        for chunk, vector in zip(chunks, vectors):
            page = int(chunk["page"])
            chunk_index = int(chunk["chunk_index"])
            metadata = {
                "record_type": "chunk",
                "doc_id": doc_id,
                "filename": filename,
                "title": title,
                "category": category,
                "page": page,
                "chunk_index": chunk_index,
                "text": chunk["text"],
                "indexed_at": indexed_at,
            }
            if source_url:
                metadata["source_url"] = source_url

            records.append({
                "id": f"doc#{doc_id}#p{page:04d}#c{chunk_index:05d}",
                "values": vector,
                "metadata": metadata,
            })

        for start in range(0, len(records), 50):
            batch = records[start:start + 50]
            last_error = None

            for attempt in range(4):
                try:
                    index.upsert(vectors=batch, namespace=KNOWLEDGE_NS)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    if attempt < 3:
                        time.sleep(2 ** attempt)

            if last_error is not None:
                raise last_error

        manifest_vector = self.embed(
            [f"{title}. {filename}. Category: {category}. BDS knowledge-base document."],
            "passage",
        )[0]

        total_characters = sum(len(page.get("text", "")) for page in pages)
        manifest_metadata = {
            "record_type": "document",
            "doc_id": doc_id,
            "filename": filename,
            "title": title,
            "category": category,
            "pages": len(pages),
            "chunks": len(chunks),
            "characters": total_characters,
            "indexed_at": indexed_at,
        }
        if source_url:
            manifest_metadata["source_url"] = source_url

        index.upsert(
            vectors=[{
                "id": self.manifest_id(doc_id),
                "values": manifest_vector,
                "metadata": manifest_metadata,
            }],
            namespace=MANIFEST_NS,
        )

        return {
            "success": True,
            "duplicate": False,
            "doc_id": doc_id,
            "filename": filename,
            "title": title,
            "category": category,
            "pages": len(pages),
            "chunks": len(chunks),
            "characters": total_characters,
        }

    def search(
        self,
        query: str,
        categories: Optional[List[str]] = None,
        top_k: int = RAG_CONTEXT_CHUNKS,
        exclude_assessment: bool = False,
    ) -> List[Dict[str, Any]]:
        index = self.connect()
        if index is None:
            return []

        vector = self.embed([query], "query")[0]

        kwargs: Dict[str, Any] = {
            "vector": vector,
            "top_k": max(RAG_CANDIDATES, top_k * (4 if exclude_assessment else 1)),
            "include_metadata": True,
            "namespace": KNOWLEDGE_NS,
        }

        if categories:
            cleaned = list(dict.fromkeys(safe_category(value) for value in categories))
            if cleaned:
                kwargs["filter"] = {"category": {"$in": cleaned}}

        result = index.query(**kwargs)
        candidates = []

        for match in _matches(result):
            metadata = _metadata(match)
            text = normalize_text(metadata.get("text", ""))
            if not text:
                continue

            semantic = _score(match)
            lexical = keyword_overlap(query, text, metadata.get("title", ""))
            combined = 0.86 * semantic + 0.14 * lexical
            assessment_like = is_assessment_like(text)

            candidates.append({
                "score": semantic,
                "combined_score": combined,
                "assessment_like": assessment_like,
                "evidence_kind": "assessment" if assessment_like else "exposition",
                "doc_id": metadata.get("doc_id", ""),
                "filename": metadata.get("filename", "Unknown source"),
                "title": metadata.get("title", ""),
                "category": metadata.get("category", "Other"),
                "page": int(metadata.get("page", 0) or 0),
                "chunk_index": int(metadata.get("chunk_index", 0) or 0),
                "text": text,
                "source_url": metadata.get("source_url", ""),
            })

        candidates.sort(key=lambda item: item["combined_score"], reverse=True)

        selected = []
        page_counts: Dict[str, int] = {}
        doc_counts: Dict[str, int] = {}

        for item in candidates:
            if item["score"] < RAG_MIN_SCORE:
                continue
            if exclude_assessment and item.get("assessment_like"):
                continue

            page_key = f'{item["doc_id"]}#{item["page"]}'
            if page_counts.get(page_key, 0) >= 2:
                continue
            if doc_counts.get(item["doc_id"], 0) >= 4:
                continue

            selected.append(item)
            page_counts[page_key] = page_counts.get(page_key, 0) + 1
            doc_counts[item["doc_id"]] = doc_counts.get(item["doc_id"], 0) + 1

            if len(selected) >= top_k:
                break

        return selected

    def list_documents(self) -> List[Dict[str, Any]]:
        index = self.connect()
        if index is None:
            return []

        ids = self._list_ids(MANIFEST_NS, "docmeta#")
        documents = []

        for start in range(0, len(ids), 100):
            batch = ids[start:start + 100]
            if not batch:
                continue

            fetched = index.fetch(ids=batch, namespace=MANIFEST_NS)
            for vector in _fetch_vectors(fetched).values():
                metadata = _metadata(vector)
                if metadata.get("record_type") == "document":
                    documents.append(metadata)

        documents.sort(key=lambda item: item.get("indexed_at", ""), reverse=True)
        return documents

    def document_pages(self, doc_id: str) -> List[Dict[str, Any]]:
        """Reconstruct readable page text from an already indexed RAG document.

        This is used to migrate older Past Papers into the structured Test Mode
        bank without asking the owner to upload the same PDF again.
        """
        index = self.connect()
        if index is None:
            return []

        ids = self._list_ids(
            KNOWLEDGE_NS,
            f"doc#{doc_id}#",
        )

        chunks: List[Dict[str, Any]] = []

        for start in range(0, len(ids), 100):
            batch = ids[start:start + 100]
            if not batch:
                continue

            fetched = index.fetch(
                ids=batch,
                namespace=KNOWLEDGE_NS,
            )

            for vector in _fetch_vectors(fetched).values():
                metadata = _metadata(vector)

                if metadata.get("record_type") != "chunk":
                    continue

                chunks.append({
                    "page": int(metadata.get("page", 0) or 0),
                    "chunk_index": int(
                        metadata.get("chunk_index", 0)
                        or 0
                    ),
                    "text": normalize_text(
                        metadata.get("text", "")
                    ),
                })

        chunks.sort(
            key=lambda item: (
                item["page"],
                item["chunk_index"],
            )
        )

        pages: Dict[int, str] = {}

        def merge_overlap(existing: str, incoming: str) -> str:
            if not existing:
                return incoming
            if not incoming:
                return existing

            max_overlap = min(
                CHUNK_OVERLAP + 120,
                len(existing),
                len(incoming),
            )

            overlap = 0
            for size in range(max_overlap, 39, -1):
                if existing[-size:] == incoming[:size]:
                    overlap = size
                    break

            if overlap:
                return existing + incoming[overlap:]

            return existing + "\n" + incoming

        for chunk in chunks:
            page = chunk["page"]
            pages[page] = merge_overlap(
                pages.get(page, ""),
                chunk["text"],
            )

        return [
            {
                "page": page,
                "text": normalize_text(text),
            }
            for page, text in sorted(pages.items())
            if normalize_text(text)
        ]


    def delete_document(self, doc_id: str, require_present: bool = True) -> Dict[str, Any]:
        index = self.connect()
        if index is None:
            raise RuntimeError("Persistent RAG is not configured.")

        if require_present and not self.manifest_exists(doc_id):
            raise KeyError("Document not found.")

        ids = self._list_ids(KNOWLEDGE_NS, f"doc#{doc_id}#")
        for start in range(0, len(ids), 1000):
            batch = ids[start:start + 1000]
            if batch:
                index.delete(ids=batch, namespace=KNOWLEDGE_NS)

        index.delete(ids=[self.manifest_id(doc_id)], namespace=MANIFEST_NS)
        return {"success": True, "doc_id": doc_id, "deleted_chunks": len(ids)}

    def status(self) -> Dict[str, Any]:
        response = {
            "configured": self.configured,
            "index_name": INDEX_NAME,
            "embedding_model": EMBED_MODEL,
            "persistent": self.configured,
        }

        if not self.configured:
            response["message"] = "Add PINECONE_API_KEY to enable persistent RAG."
            return response

        try:
            index = self.connect()
            stats = index.describe_index_stats()
            if isinstance(stats, dict):
                count = stats.get("total_vector_count", 0) or 0
            else:
                count = getattr(stats, "total_vector_count", 0) or 0
            response["total_vector_count"] = int(count)
        except Exception as exc:
            response["warning"] = str(exc)

        return response


def build_context(sources: List[Dict[str, Any]]) -> str:
    if not sources:
        return "No relevant knowledge-base source was retrieved."

    blocks = []
    for number, source in enumerate(sources, start=1):
        blocks.append(
            f"[S{number}]\n"
            f"Source: {source.get('filename', 'Unknown source')}\n"
            f"Category: {source.get('category', 'Other')}\n"
            f"Page: {source.get('page', 0)}\n"
            f"Excerpt:\n{source.get('text', '')}"
        )
    return "\n\n".join(blocks)


def public_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []

    for number, source in enumerate(sources, start=1):
        snippet = re.sub(r"\s+", " ", source.get("text", "")).strip()
        if len(snippet) > 240:
            snippet = snippet[:237].rstrip() + "..."

        item = {
            "label": f"S{number}",
            "doc_id": source.get("doc_id", ""),
            "filename": source.get("filename", "Unknown source"),
            "title": source.get("title", ""),
            "category": source.get("category", "Other"),
            "page": source.get("page", 0),
            "score": round(float(source.get("score", 0.0)), 4),
            "evidence_kind": source.get("evidence_kind", "exposition"),
            "snippet": snippet,
        }

        if source.get("source_url"):
            item["source_url"] = source["source_url"]

        result.append(item)

    return result


rag_store = RagStore()
