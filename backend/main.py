from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from io import BytesIO
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from google import genai
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
from typing import Any, Dict, List, Optional
import os
import re
import time
import hmac
from threading import Lock

import pytesseract
import ollama

from rag import (
    RAG_CONTEXT_CHUNKS,
    build_context,
    chunk_pages,
    document_id,
    is_assessment_like,
    is_strict_source_request,
    normalize_text,
    public_sources,
    rag_store,
    safe_category,
)


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

DENTORA_ADMIN_KEY = os.getenv("DENTORA_ADMIN_KEY", "").strip()
DENTORA_BETA_ACCESS_CODE = os.getenv("DENTORA_BETA_ACCESS_CODE", "").strip()
MAX_WEB_INGEST_MB = int(os.getenv("MAX_WEB_INGEST_MB", "45"))
MAX_WEB_OCR_PAGES = int(os.getenv("MAX_WEB_OCR_PAGES", "30"))

# Public beta safety defaults. These can be overridden in Render.
MAX_TEMP_PDF_MB = int(os.getenv("MAX_TEMP_PDF_MB", "20"))
MAX_TEMP_PDF_PAGES = int(os.getenv("MAX_TEMP_PDF_PAGES", "500"))
SESSION_PDF_TTL_SECONDS = int(os.getenv("SESSION_PDF_TTL_SECONDS", str(4 * 60 * 60)))
CHAT_RATE_LIMIT = int(os.getenv("CHAT_RATE_LIMIT", "30"))
CHAT_RATE_WINDOW_SECONDS = int(os.getenv("CHAT_RATE_WINDOW_SECONDS", "600"))
PDF_RATE_LIMIT = int(os.getenv("PDF_RATE_LIMIT", "3"))
PDF_RATE_WINDOW_SECONDS = int(os.getenv("PDF_RATE_WINDOW_SECONDS", "3600"))

_default_origins = (
    "https://drmes-dev.github.io,"
    "http://localhost:5500,"
    "http://127.0.0.1:5500,"
    "http://localhost:8000"
)
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv("DENTORA_ALLOWED_ORIGINS", _default_origins).split(",")
    if origin.strip()
]


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="Dentora API",
    version="2.0.0",
    description="Dentora BDS study assistant with persistent evidence-grounded RAG.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["Content-Type", "X-Dentora-Admin-Key", "X-Dentora-Beta-Key"],
)


# =========================================================
# AI CLIENTS
# =========================================================

gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

qwen_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
)


# =========================================================
# REQUEST MODELS
# =========================================================

class ChatRequest(BaseModel):
    message: str
    mode: str = "study"
    session_id: str = "default"
    use_library: bool = True
    categories: Optional[List[str]] = None


class RagSearchRequest(BaseModel):
    query: str
    categories: Optional[List[str]] = None
    top_k: int = Field(default=7, ge=1, le=20)


class FeedbackRequest(BaseModel):
    reason: str = Field(min_length=2, max_length=2000)
    question: str = Field(default="", max_length=4000)
    answer: str = Field(default="", max_length=12000)
    mode: str = Field(default="", max_length=40)
    source_summary: str = Field(default="", max_length=4000)


# =========================================================
# TEMPORARY PDF SESSION MEMORY
# =========================================================
#
# This is intentionally temporary. Persistent knowledge is
# stored in Pinecone. Browser sessions receive separate IDs
# so one visitor's PDF is not mixed with another visitor's PDF.

session_pdfs: Dict[str, Dict[str, Any]] = {}

_rate_events: Dict[str, List[float]] = {}
_rate_lock = Lock()

feedback_events: List[Dict[str, Any]] = []
_feedback_lock = Lock()


def _client_ip(request: Request) -> str:
    # Render forwards the original client address through X-Forwarded-For.
    forwarded = request.headers.get("x-forwarded-for", "")
    if forwarded:
        return forwarded.split(",", 1)[0].strip()[:80]

    if request.client and request.client.host:
        return request.client.host[:80]

    return "unknown"


def enforce_rate_limit(
    request: Request,
    bucket: str,
    limit: int,
    window_seconds: int,
):
    """Small in-memory beta limiter.

    This protects a single Render instance from accidental/obvious abuse.
    It intentionally does not pretend to be a distributed production limiter.
    """
    now = time.time()
    key = f"{bucket}:{_client_ip(request)}"

    with _rate_lock:
        recent = [
            timestamp
            for timestamp in _rate_events.get(key, [])
            if now - timestamp < window_seconds
        ]

        if len(recent) >= limit:
            retry_after = max(
                1,
                int(window_seconds - (now - recent[0])),
            )
            _rate_events[key] = recent
            raise HTTPException(
                status_code=429,
                detail="Too many requests. Please wait before trying again.",
                headers={"Retry-After": str(retry_after)},
            )

        recent.append(now)
        _rate_events[key] = recent

        # Keep the limiter bounded on a long-running instance.
        if len(_rate_events) > 5000:
            stale_keys = [
                item_key
                for item_key, timestamps in _rate_events.items()
                if not timestamps or now - timestamps[-1] > max(
                    CHAT_RATE_WINDOW_SECONDS,
                    PDF_RATE_WINDOW_SECONDS,
                )
            ]
            for item_key in stale_keys[:1000]:
                _rate_events.pop(item_key, None)


def purge_expired_session_pdfs():
    now = time.time()
    expired = [
        session_id
        for session_id, item in session_pdfs.items()
        if now - float(item.get("uploaded_at", 0) or 0)
        > SESSION_PDF_TTL_SECONDS
    ]
    for session_id in expired:
        session_pdfs.pop(session_id, None)


# =========================================================
# PDF HELPERS
# =========================================================

def extract_pdf_pages(
    contents: bytes,
    allow_ocr: bool = True,
    web_mode: bool = True,
) -> Dict[str, Any]:
    reader = PdfReader(BytesIO(contents))
    pages = []
    scanned_pages = []

    for number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")

        if len(text) < 40:
            scanned_pages.append(number)

        pages.append({
            "page": number,
            "text": text,
        })

    ocr_used = False
    ocr_failed = False

    if (
        allow_ocr
        and scanned_pages
        and (not web_mode or len(reader.pages) <= MAX_WEB_OCR_PAGES)
    ):
        try:
            images = convert_from_bytes(contents, dpi=180)

            for page_number in scanned_pages:
                ocr_text = normalize_text(
                    pytesseract.image_to_string(images[page_number - 1])
                )
                if ocr_text:
                    pages[page_number - 1]["text"] = ocr_text

            ocr_used = True

        except Exception as exc:
            print("OCR fallback error:", exc)
            ocr_failed = True

    unresolved = [
        page["page"]
        for page in pages
        if len(normalize_text(page.get("text", ""))) < 40
    ]

    return {
        "pages": pages,
        "page_count": len(reader.pages),
        "ocr_used": ocr_used,
        "ocr_failed": ocr_failed,
        "unresolved_scanned_pages": unresolved,
    }


def keyword_score(query: str, text: str) -> float:
    words = {
        word
        for word in re.findall(r"[a-zA-Z0-9]+", str(query).lower())
        if len(word) > 2
    }
    if not words:
        return 0.0

    haystack = str(text).lower()
    hits = sum(1 for word in words if word in haystack)
    return hits / len(words)


def find_relevant_session_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    max_chunks: int = 4,
) -> List[Dict[str, Any]]:
    ranked = []

    for chunk in chunks:
        score = keyword_score(query, chunk.get("text", ""))
        if score > 0:
            ranked.append((score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in ranked[:max_chunks]]


# =========================================================
# SECURITY
# =========================================================

def require_admin(supplied_key: Optional[str]):
    if not DENTORA_ADMIN_KEY:
        raise HTTPException(
            status_code=503,
            detail="DENTORA_ADMIN_KEY is not configured on the server.",
        )

    if not supplied_key or not hmac.compare_digest(
        supplied_key,
        DENTORA_ADMIN_KEY,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid admin key.",
        )


def require_beta_access(supplied_key: Optional[str]):
    """Require the separate student beta access code.

    Never reuse DENTORA_ADMIN_KEY here. The beta code may be shared with
    invited students; the admin key must remain private.
    """
    if not DENTORA_BETA_ACCESS_CODE:
        raise HTTPException(
            status_code=503,
            detail=(
                "Dentora beta access is not configured yet. "
                "The administrator needs to set DENTORA_BETA_ACCESS_CODE."
            ),
        )

    if not supplied_key or not hmac.compare_digest(
        supplied_key,
        DENTORA_BETA_ACCESS_CODE,
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid beta access code.",
        )


# =========================================================
# AI GENERATION
# =========================================================

def generate_with_fallback(prompt: str) -> Dict[str, str]:
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )
        if response and response.text:
            return {
                "response": response.text,
                "provider": "Gemini",
            }
    except Exception as exc:
        print("Gemini error:", exc)

    try:
        response = groq_client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=[{
                "role": "user",
                "content": prompt,
            }],
        )
        text = response.choices[0].message.content
        if text:
            return {
                "response": text,
                "provider": "Groq",
            }
    except Exception as exc:
        print("Groq error:", exc)

    try:
        response = qwen_client.chat.completions.create(
            model="qwen-plus",
            messages=[{
                "role": "user",
                "content": prompt,
            }],
        )
        text = response.choices[0].message.content
        if text:
            return {
                "response": text,
                "provider": "Qwen API",
            }
    except Exception as exc:
        print("Qwen API error:", exc)

    try:
        response = ollama.chat(
            model="qwen3:8b",
            messages=[{
                "role": "user",
                "content": prompt,
            }],
        )
        text = response.get("message", {}).get("content", "")
        if text:
            return {
                "response": text,
                "provider": "Local Qwen",
            }
    except Exception as exc:
        print("Ollama error:", exc)

    return {
        "response": "Dentora couldn't connect to the AI service. Please try again.",
        "provider": "Error",
    }



def verify_source_grounding(
    *,
    question: str,
    draft: str,
    knowledge_context: str,
) -> Dict[str, str]:
    """Second-pass rewrite for strict source-bound questions.

    The verifier is deliberately conservative: unsupported useful background
    should be removed rather than blended into textbook evidence.
    """
    prompt = f"""
You are Dentora's grounding verifier.

TASK
----
Rewrite the draft answer so it is strictly faithful to the retrieved source
excerpts. Return ONLY the corrected answer.

STUDENT QUESTION
----------------
{question}

RETRIEVED SOURCE EXCERPTS
-------------------------
{knowledge_context}

DRAFT ANSWER TO VERIFY
----------------------
{draft}

STRICT VERIFICATION RULES
-------------------------
1. Keep a factual claim only if it is directly supported by the retrieved
   excerpt cited for that claim.
2. Do not add general dental knowledge, even if it is correct, unless the
   student's question explicitly asks for outside/background knowledge.
3. Do not infer composition, mechanism, clinical effects, percentages,
   chemical actions, recommendations, or protocol variants beyond what the
   excerpts explicitly state.
4. A figure showing that a sequence worked supports "the figure shows/reports
   successful removal with that sequence"; it does not automatically prove
   that the sequence is universally recommended or superior.
5. A passage saying particles are "primarily inorganic" does not by itself
   support exact percentages, hydroxyapatite composition, or a description
   of the organic fraction.
6. If a requested point is not directly supported, say:
   "This is not clearly covered in the retrieved textbook excerpts."
7. Preserve valid [S1], [S2], etc. citations. Never invent a source label,
   page number, quotation, or reference.
8. Do not attach a citation to a claim that the cited excerpt does not support.
9. Only use quotation marks for wording that appears verbatim in the supplied
   excerpt. Otherwise paraphrase without quotation marks.
10. Review questions, MCQ options, distractors, and true/false statements are
    not factual evidence unless the supplied excerpt clearly establishes their
    correctness.
11. Keep the answer useful for a final-year BDS student, but evidence fidelity
    is more important than completeness.
12. Before returning the answer, inspect every sentence and table row for
    unsupported details and remove or narrow them.

Return the corrected answer only.
"""
    return generate_with_fallback(prompt)


# =========================================================
# HEALTH
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Dentora backend is running.",
        "version": "2.1.0-beta-safety",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "Dentora API",
        "rag_configured": rag_store.configured,
        "beta_access_required": True,
        "beta_access_configured": bool(DENTORA_BETA_ACCESS_CODE),
    }


@app.post("/beta/verify")
def beta_verify(
    request: Request,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    enforce_rate_limit(
        request,
        "beta-verify",
        12,
        15 * 60,
    )
    require_beta_access(x_dentora_beta_key)
    return {"ok": True, "access": "beta"}


@app.get("/rag/status")
def rag_status():
    return rag_store.status()


# =========================================================
# TEMPORARY PDF TUTOR
# =========================================================

@app.post("/upload-pdf")
async def upload_pdf(
    request: Request,
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    require_beta_access(x_dentora_beta_key)

    enforce_rate_limit(
        request,
        "pdf-upload",
        PDF_RATE_LIMIT,
        PDF_RATE_WINDOW_SECONDS,
    )
    purge_expired_session_pdfs()

    filename = file.filename or "uploaded.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    try:
        max_bytes = MAX_TEMP_PDF_MB * 1024 * 1024
        contents = await file.read(max_bytes + 1)

        if len(contents) > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"PDF Tutor accepts files up to {MAX_TEMP_PDF_MB} MB. "
                    "Use a smaller chapter or compressed PDF."
                ),
            )

        try:
            page_count = len(PdfReader(BytesIO(contents)).pages)
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="The uploaded file could not be read as a valid PDF.",
            ) from exc

        if page_count > MAX_TEMP_PDF_PAGES:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"PDF Tutor accepts up to {MAX_TEMP_PDF_PAGES} pages "
                    "per temporary upload."
                ),
            )

        extraction = extract_pdf_pages(contents, allow_ocr=True, web_mode=True)
        chunks = chunk_pages(extraction["pages"])

        if not chunks:
            return {
                "success": False,
                "message": (
                    "No readable text was extracted. "
                    "For a large scanned PDF, use Dentora's local library indexer."
                ),
                "pages": extraction["page_count"],
                "unresolved_scanned_pages": extraction["unresolved_scanned_pages"],
            }

        session_id = (session_id.strip() or "default")[:120]

        session_pdfs[session_id] = {
            "name": filename,
            "chunks": chunks,
            "pages": extraction["page_count"],
            "uploaded_at": time.time(),
        }

        if len(session_pdfs) > 50:
            oldest = sorted(
                session_pdfs.items(),
                key=lambda item: item[1].get("uploaded_at", 0),
            )[:10]

            for old_key, _ in oldest:
                session_pdfs.pop(old_key, None)

        return {
            "success": True,
            "filename": filename,
            "pages": extraction["page_count"],
            "chunks": len(chunks),
            "ocr_used": extraction["ocr_used"],
            "unresolved_scanned_pages": extraction["unresolved_scanned_pages"],
            "message": "PDF is ready for this chat session.",
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("PDF upload error:", exc)
        raise HTTPException(
            status_code=500,
            detail="Dentora could not process this PDF.",
        ) from exc


@app.delete("/upload-pdf/{session_id}")
def remove_session_pdf(
    session_id: str,
    request: Request,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    require_beta_access(x_dentora_beta_key)

    enforce_rate_limit(
        request,
        "pdf-remove",
        20,
        PDF_RATE_WINDOW_SECONDS,
    )
    session_id = (session_id.strip() or "default")[:120]
    removed = session_pdfs.pop(session_id, None) is not None
    return {"success": True, "removed": removed}


# =========================================================
# PERSISTENT LIBRARY INGESTION
# =========================================================

@app.post("/rag/ingest-pdf")
async def rag_ingest_pdf(
    file: UploadFile = File(...),
    category: str = Form("Other"),
    title: str = Form(""),
    source_url: str = Form(""),
    replace_existing: bool = Form(False),
    x_dentora_admin_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Admin-Key",
    ),
):
    require_admin(x_dentora_admin_key)

    if not rag_store.configured:
        raise HTTPException(
            status_code=503,
            detail="Persistent RAG is not configured yet.",
        )

    filename = file.filename or "document.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    contents = await file.read()

    if len(contents) > MAX_WEB_INGEST_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Web ingestion is limited to {MAX_WEB_INGEST_MB} MB per PDF. "
                "Use tools/index_library.py for large files."
            ),
        )

    extraction = extract_pdf_pages(contents, allow_ocr=True, web_mode=True)
    unresolved = extraction["unresolved_scanned_pages"]

    if (
        unresolved
        and len(unresolved)
        > max(3, int(extraction["page_count"] * 0.20))
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "This PDF is mostly scanned and needs local OCR "
                    "before persistent indexing."
                ),
                "unresolved_scanned_pages": unresolved,
                "recommended_command": (
                    f'python tools/index_library.py "PATH_TO_PDF" '
                    f'--category "{safe_category(category)}"'
                ),
            },
        )

    resolved_title = title.strip() or os.path.splitext(filename)[0]

    try:
        result = rag_store.index_pages(
            doc_id=document_id(contents),
            filename=filename,
            title=resolved_title,
            category=category,
            pages=extraction["pages"],
            source_url=source_url.strip(),
            replace_existing=replace_existing,
        )
        result["ocr_used"] = extraction["ocr_used"]
        result["unresolved_scanned_pages"] = unresolved
        return result

    except Exception as exc:
        print("RAG ingestion error:", exc)
        raise HTTPException(status_code=500, detail=str(exc))


# =========================================================
# LIBRARY / SEARCH / DELETE
# =========================================================

@app.get("/rag/library")
def rag_library(
    x_dentora_admin_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Admin-Key",
    ),
):
    require_admin(x_dentora_admin_key)

    if not rag_store.configured:
        return {
            "configured": False,
            "documents": [],
        }

    try:
        documents = rag_store.list_documents()
        return {
            "configured": True,
            "count": len(documents),
            "documents": documents,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/rag/search")
def rag_search(
    request: RagSearchRequest,
    x_dentora_admin_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Admin-Key",
    ),
):
    require_admin(x_dentora_admin_key)

    if not rag_store.configured:
        return {
            "configured": False,
            "results": [],
        }

    try:
        results = rag_store.search(
            request.query,
            categories=request.categories,
            top_k=request.top_k,
        )
        return {
            "configured": True,
            "query": request.query,
            "results": public_sources(results),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/rag/document/{doc_id}")
def rag_delete_document(
    doc_id: str,
    x_dentora_admin_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Admin-Key",
    ),
):
    require_admin(x_dentora_admin_key)

    if not rag_store.configured:
        raise HTTPException(
            status_code=503,
            detail="Persistent RAG is not configured yet.",
        )

    try:
        return rag_store.delete_document(doc_id)
    except KeyError:
        raise HTTPException(status_code=404, detail="Document not found.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# =========================================================
# FEEDBACK
# =========================================================

@app.post("/feedback")
def submit_feedback(
    feedback: FeedbackRequest,
    request: Request,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    require_beta_access(x_dentora_beta_key)
    enforce_rate_limit(
        request,
        "feedback",
        10,
        60 * 60,
    )

    item = {
        "feedback_id": f"fb-{int(time.time() * 1000)}",
        "created_at": time.time(),
        "reason": feedback.reason.strip(),
        "question": feedback.question.strip(),
        "answer": feedback.answer.strip(),
        "mode": feedback.mode.strip(),
        "source_summary": feedback.source_summary.strip(),
    }

    with _feedback_lock:
        feedback_events.append(item)
        if len(feedback_events) > 500:
            del feedback_events[:-500]

    print(
        "Dentora feedback:",
        item["feedback_id"],
        item["reason"][:160],
    )

    return {
        "success": True,
        "feedback_id": item["feedback_id"],
        "message": "Feedback received.",
    }


@app.get("/feedback")
def list_feedback(
    x_dentora_admin_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Admin-Key",
    ),
):
    require_admin(x_dentora_admin_key)

    with _feedback_lock:
        return {
            "count": len(feedback_events),
            "feedback": list(reversed(feedback_events[-200:])),
        }


# =========================================================
# CHAT
# =========================================================

@app.post("/chat")
def chat(
    request: ChatRequest,
    http_request: Request,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    require_beta_access(x_dentora_beta_key)

    enforce_rate_limit(
        http_request,
        "chat",
        CHAT_RATE_LIMIT,
        CHAT_RATE_WINDOW_SECONDS,
    )
    purge_expired_session_pdfs()

    message = request.message.strip()
    mode_key = request.mode.strip().lower()
    exclude_assessment = mode_key not in {"mcq", "mcq mode", "quiz"}
    strict_source = is_strict_source_request(message, request.mode)

    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    retrieved_sources: List[Dict[str, Any]] = []
    rag_warning = ""

    # Temporary PDF Tutor source
    session_pdf = session_pdfs.get(request.session_id)

    if session_pdf:
        session_added = 0
        for item in find_relevant_session_chunks(
            message,
            session_pdf.get("chunks", []),
            max_chunks=12 if exclude_assessment else 4,
        ):
            if exclude_assessment and is_assessment_like(item.get("text", "")):
                continue

            retrieved_sources.append({
                "score": 1.0,
                "combined_score": 1.0,
                "doc_id": f"session:{request.session_id}",
                "filename": session_pdf.get("name", "Uploaded PDF"),
                "title": session_pdf.get("name", "Uploaded PDF"),
                "category": "Uploaded PDF",
                "page": item.get("page", 0),
                "chunk_index": item.get("chunk_index", 0),
                "text": item.get("text", ""),
                "source_url": "",
                "evidence_kind": "exposition",
            })
            session_added += 1
            if session_added >= 4:
                break

    # Persistent library
    if request.use_library and rag_store.configured:
        try:
            library_sources = rag_store.search(
                message,
                categories=request.categories,
                top_k=RAG_CONTEXT_CHUNKS,
                exclude_assessment=exclude_assessment,
            )

            existing = {
                (
                    source.get("filename", ""),
                    source.get("page", 0),
                    source.get("text", "")[:80],
                )
                for source in retrieved_sources
            }

            for source in library_sources:
                key = (
                    source.get("filename", ""),
                    source.get("page", 0),
                    source.get("text", "")[:80],
                )
                if key not in existing:
                    retrieved_sources.append(source)
                    existing.add(key)

                if len(retrieved_sources) >= RAG_CONTEXT_CHUNKS:
                    break

        except Exception as exc:
            print("RAG retrieval error:", exc)
            rag_warning = str(exc)

    rag_used = bool(retrieved_sources)
    knowledge_context = build_context(retrieved_sources)

    if rag_used:
        knowledge_rules = """
DENTORA KNOWLEDGE-BASE RULES
----------------------------
The raw PDF excerpts below are evidence candidates, NOT pre-verified facts.

1. Answer from explanatory textbook prose, supported tables, and figure captions.
2. WARNING: textbook review questions, MCQ options, true/false statements,
   mock answers, and distractors may be present in extracted page text.
   NEVER repeat an answer option as a factual textbook claim just because
   that option appears in a retrieved passage. Use it as evidence only when
   the answer key or accompanying explanatory prose establishes its validity.
   If uncertain, do not use the option as evidence.
3. Distinguish what the textbook says, what a question merely asks, and
   what you independently infer. Do not present an inference as a quotation.
4. Cite each claim with its corresponding [S1], [S2], etc. label. Use only
   the exact source label and PDF page number printed beside that excerpt.
   PDF page numbers and the textbook printed page can differ. Never invent
   or transplant page numbers between sources.
5. If the student asks for one named textbook, author, or excludes a book,
   respect that restriction EVEN IF excerpts from other books were retrieved.
   Do not cite or rely on an excluded book.
6. Only put text inside quotation marks if it is verbatim in the excerpt.
   Otherwise clearly paraphrase it, with a citation if supported.
7. Before answering, check whether each substantive sourced claim follows
   from explanatory evidence rather than a question, distractor, or heading.
8. If the excerpts do not clearly support a requested point, explicitly say:
   "This is not clearly covered in your uploaded resources."
   Do not use an unrelated passage to justify a missing answer.
9. Only after clearly identifying the gap may you provide established
   supplementary BDS-level knowledge under:
   "Additional background knowledge — not directly from your uploaded resources."
   Never attach an uploaded-source citation to supplementary knowledge.
10. STRICT GROUNDING: if the student's wording asks "according to" an uploaded
    textbook/resource, requests textbook page references, or explicitly restricts
    the answer to a named source, keep the main answer source-only. Do not add
    precise percentages, measurements, timings, sequences, mechanisms, or other
    details unless they are explicitly supported by a retrieved excerpt. If a
    requested point is missing, say that it is not clearly covered. Do not fill
    it with uncited background unless the student explicitly asks for background.
11. Never mix sourced and unsourced claims in the same sentence, table row, or
    viva-summary statement. A source label applies only to the exact claim that
    the excerpt supports. Split unsupported background into a separate clearly
    labelled section with no uploaded-source citation.
12. Final summaries and viva answers must preserve the same evidence boundaries
    as the detailed answer. Do not reintroduce unsupported details in a summary.
13. Do not broaden a protocol or sequence beyond the retrieved evidence. For
    example, if the source supports A followed by B, do not add "or vice versa"
    unless another retrieved excerpt explicitly supports that alternative.
14. If sources disagree, identify the difference rather than silently
    reconciling them. Use well-formed Markdown tables when helpful.
"""
    else:
        knowledge_rules = """
DENTORA KNOWLEDGE-BASE RULES
----------------------------
No relevant uploaded/library source was retrieved.

Start by stating:
"This is not clearly covered in your uploaded resources."

You may then answer from established BDS-level knowledge under:
"Additional background knowledge — not directly from your uploaded resources."

Do not imply that background knowledge came from the student's library.
"""

    prompt = f"""
You are Dentora by Ehsan — an evidence-grounded BDS study assistant.

STUDENT LEVEL
-------------
Final-year BDS.

CURRENT MODE
------------
{request.mode}

STUDENT'S QUESTION
------------------
{message}

{knowledge_rules}

RETRIEVED SOURCE EXCERPTS
-------------------------
{knowledge_context}

MODE-SPECIFIC INSTRUCTIONS
--------------------------
- Study Chat: explain clearly, clinically, and exam-relevantly.
- MCQ Mode: create or explain one-best-answer questions with concise reasoning.
- Viva Mode: prioritize short examiner-ready answers, then key follow-up points.
- OSCE Mode: structure answers around station steps, identification, findings, interpretation, safety, and examiner prompts.
- PDF Tutor: stay especially strict to uploaded/retrieved material.

GENERAL PRESENTATION
--------------------
- Answer at final-year BDS level.
- Use clear headings, bullets, and Markdown tables when useful.
- Keep facts precise.
- Do not fabricate references.
- Do not unnecessarily repeat the question.

Answer now.
"""

    generated = generate_with_fallback(prompt)

    if (
        strict_source
        and rag_used
        and generated.get("provider") != "Error"
        and generated.get("response")
    ):
        verified = verify_source_grounding(
            question=message,
            draft=generated["response"],
            knowledge_context=knowledge_context,
        )
        if verified.get("provider") != "Error" and verified.get("response"):
            generated["response"] = verified["response"]
            generated["grounding_verified"] = True
            generated["verification_provider"] = verified.get("provider", "")
        else:
            generated["grounding_verified"] = False
    else:
        generated["grounding_verified"] = False

    generated["rag_used"] = rag_used
    generated["rag_configured"] = rag_store.configured
    generated["sources"] = public_sources(retrieved_sources)

    if rag_warning:
        generated["rag_warning"] = rag_warning

    return generated
