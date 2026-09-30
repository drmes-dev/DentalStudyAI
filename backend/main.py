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
import json
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
from test_engine import past_paper_store


# =========================================================
# ENVIRONMENT
# =========================================================

load_dotenv()

DENTORA_ADMIN_KEY = os.getenv("DENTORA_ADMIN_KEY", "").strip()
DENTORA_BETA_ACCESS_CODE = os.getenv("DENTORA_BETA_ACCESS_CODE", "").strip()
DENTORA_OWNER_ACCESS_CODE = os.getenv("DENTORA_OWNER_ACCESS_CODE", "").strip()
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
MAX_VOICE_AUDIO_MB = int(os.getenv("MAX_VOICE_AUDIO_MB", "12"))
VOICE_RATE_LIMIT = int(os.getenv("VOICE_RATE_LIMIT", "20"))
VOICE_RATE_WINDOW_SECONDS = int(os.getenv("VOICE_RATE_WINDOW_SECONDS", "3600"))

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


class TestStartRequest(BaseModel):
    count: int = Field(default=20, ge=1, le=100)
    subject: Optional[str] = Field(default=None, max_length=160)
    year: Optional[str] = Field(default=None, max_length=40)
    topic: Optional[str] = Field(default=None, max_length=160)
    repeated_only: bool = False
    weak_topics: Optional[List[str]] = None


class TestGradeRequest(BaseModel):
    responses: List[Dict[str, str]] = Field(default_factory=list)


class ImportRagPastPaperRequest(BaseModel):
    subject: Optional[str] = Field(default=None, max_length=160)
    year: Optional[str] = Field(default=None, max_length=40)
    title: Optional[str] = Field(default=None, max_length=300)


class ExistingPaperImportRequest(BaseModel):
    subject: Optional[str] = Field(default=None, max_length=160)
    year: Optional[str] = Field(default=None, max_length=40)
    title: Optional[str] = Field(default=None, max_length=300)


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


def access_role(supplied_key: Optional[str]) -> Optional[str]:
    """Return owner/beta for a valid app-access key, otherwise None."""
    if not supplied_key:
        return None

    if (
        DENTORA_OWNER_ACCESS_CODE
        and hmac.compare_digest(
            supplied_key,
            DENTORA_OWNER_ACCESS_CODE,
        )
    ):
        return "owner"

    if (
        DENTORA_BETA_ACCESS_CODE
        and hmac.compare_digest(
            supplied_key,
            DENTORA_BETA_ACCESS_CODE,
        )
    ):
        return "beta"

    return None


def require_beta_access(supplied_key: Optional[str]) -> str:
    """Allow either the private owner key or the invited-student beta key.

    DENTORA_ADMIN_KEY remains separate and is never accepted here.
    """
    if not (
        DENTORA_BETA_ACCESS_CODE
        or DENTORA_OWNER_ACCESS_CODE
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Dentora access is not configured yet. "
                "Set DENTORA_OWNER_ACCESS_CODE and/or "
                "DENTORA_BETA_ACCESS_CODE."
            ),
        )

    role = access_role(supplied_key)

    if role is None:
        raise HTTPException(
            status_code=401,
            detail="Invalid Dentora access code.",
        )

    return role


def require_owner_access(supplied_key: Optional[str]):
    if not DENTORA_OWNER_ACCESS_CODE:
        raise HTTPException(
            status_code=503,
            detail="DENTORA_OWNER_ACCESS_CODE is not configured.",
        )

    if (
        not supplied_key
        or not hmac.compare_digest(
            supplied_key,
            DENTORA_OWNER_ACCESS_CODE,
        )
    ):
        raise HTTPException(
            status_code=403,
            detail="Owner access is required for this action.",
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


def _json_payload(text: str, expected: str = "array"):
    value = str(text or "").strip()
    value = re.sub(
        r"^\s*```(?:json)?\s*",
        "",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s*```\s*$", "", value)

    if expected == "object":
        start = value.find("{")
        end = value.rfind("}")
    else:
        start = value.find("[")
        end = value.rfind("]")

    if start < 0 or end <= start:
        raise ValueError("Model did not return valid JSON.")

    return json.loads(value[start:end + 1])


def _past_paper_batches(
    pages: List[Dict[str, Any]],
    max_chars: int = 14000,
) -> List[str]:
    batches = []
    current = []
    length = 0

    for page in pages:
        text = normalize_text(page.get("text", ""))
        if not text:
            continue

        block = (
            f"--- PDF PAGE {int(page.get('page', 0))} ---\n"
            f"{text}"
        )

        if current and length + len(block) > max_chars:
            batches.append("\n\n".join(current))
            current = []
            length = 0

        current.append(block)
        length += len(block)

    if current:
        batches.append("\n\n".join(current))

    return batches


def parse_past_paper_questions(
    *,
    pages: List[Dict[str, Any]],
    subject: str,
) -> List[Dict[str, Any]]:
    questions: List[Dict[str, Any]] = []
    seen = set()

    for batch in _past_paper_batches(pages):
        prompt = f"""
You are structuring a BDS past examination paper into a question bank.

SUBJECT
-------
{subject}

SOURCE PAPER TEXT
-----------------
{batch}

IMPORTANT OCR NOTE
------------------
This text may come from scanned exam pages and OCR. Detect MCQs even when
line breaks are messy, option labels are separated from option text, or
characters such as A/B/C/D, 1/2/3/4, brackets, periods, and parentheses are
imperfectly recognized. Repair only obvious OCR spacing/line-break damage;
do not invent missing question content.

OUTPUT
------
Return ONLY a JSON array. Each item must have exactly these fields:
- question_number
- stem
- options
- question_type
- topic
- subtopic
- page
- marks
- provided_answer
- suggested_answer
- suggested_confidence

RULES
-----
1. Preserve the actual question wording from the paper. Do not rewrite it.
2. Preserve MCQ option wording. Use an object such as
   {{"A":"...","B":"...","C":"...","D":"..."}}.
3. If options are not present, use an empty object.
4. "provided_answer" must be blank unless an answer/key is explicitly visible
   in the supplied paper text. Never invent a paper answer key.
5. For an MCQ, "suggested_answer" may contain the single best option label
   (A/B/C/D/etc.) using standard undergraduate dental knowledge. Leave it
   blank if the OCR/options are too damaged or genuinely ambiguous.
6. "suggested_confidence" must be high, moderate, or low for a suggested
   answer, otherwise blank.
7. "page" is the PDF page number indicated by the nearest PDF PAGE marker.
8. Classify topic and subtopic concisely for BDS revision.
9. question_type should be one of:
   mcq, true_false, emq, short_answer, essay, osce, other.
10. Do not convert headings, instructions, roll numbers, marks tables, or
   answer-key labels into questions.
11. For scanned/OCR MCQs, join wrapped lines belonging to the same stem or
   option. Accept recognizable option formats such as A., A), (A), a., 1.,
   1), or (1), and preserve their labels in the options object.
12. If a question is incomplete in this excerpt, include it only when its stem
    and answer choices can be reconstructed directly from the supplied text.
13. Return JSON only. No commentary and no Markdown.
"""
        generated = generate_with_fallback(prompt)

        if generated.get("provider") == "Error":
            raise RuntimeError(
                "Dentora could not structure the past paper because no AI "
                "provider was available."
            )

        parsed = _json_payload(
            generated.get("response", ""),
            "array",
        )

        if not isinstance(parsed, list):
            raise ValueError("Past-paper parser returned a non-list payload.")

        for raw in parsed:
            if not isinstance(raw, dict):
                continue

            stem = normalize_text(raw.get("stem", ""))
            if len(stem) < 8:
                continue

            key = re.sub(
                r"[^a-z0-9]+",
                " ",
                stem.lower(),
            ).strip()

            if key in seen:
                continue

            seen.add(key)
            questions.append(raw)

            if len(questions) >= 300:
                return questions

    return questions


def canonicalize_past_paper_topics(
    questions: List[Dict[str, Any]],
    subject: str,
) -> List[Dict[str, Any]]:
    raw_topics = sorted({
        normalize_text(item.get("topic", ""))
        for item in questions
        if normalize_text(item.get("topic", ""))
    })

    if not raw_topics:
        return questions

    existing_topics: List[str] = []

    try:
        catalog = past_paper_store.catalog()
        existing_topics = [
            item.get("name", "")
            for item in catalog.get("topics", [])
            if (
                str(item.get("subject", "")).strip().lower()
                == str(subject).strip().lower()
                and item.get("name")
            )
        ][:80]
    except Exception as exc:
        print("Topic catalog lookup warning:", exc)

    prompt = f"""
You are normalizing topic labels for a BDS past-paper question bank.

SUBJECT
-------
{subject}

RAW TOPIC LABELS FROM THIS PAPER
--------------------------------
{json.dumps(raw_topics, ensure_ascii=False)}

EXISTING CANONICAL LABELS ALREADY USED FOR THIS SUBJECT
-------------------------------------------------------
{json.dumps(existing_topics, ensure_ascii=False)}

Return ONLY one JSON object mapping every raw topic label to one canonical
topic label.

RULES
-----
1. If an existing canonical label means the same thing, reuse it exactly.
2. Otherwise create a concise standard BDS topic label, normally 2-5 words.
3. Merge only genuinely equivalent labels; do not collapse distinct topics.
4. Do not change question wording, answers, or subtopics.
5. Return JSON only.
"""

    generated = generate_with_fallback(prompt)

    if generated.get("provider") == "Error":
        return questions

    try:
        mapping = _json_payload(
            generated.get("response", ""),
            "object",
        )
    except Exception as exc:
        print("Topic normalization warning:", exc)
        return questions

    if not isinstance(mapping, dict):
        return questions

    for item in questions:
        raw = normalize_text(item.get("topic", ""))
        canonical = normalize_text(
            mapping.get(raw, raw)
        )[:160]

        if canonical:
            item["topic"] = canonical

    return questions


def canonical_bds_subject(
    *values: str,
) -> str:
    text = " ".join(
        normalize_text(value)
        for value in values
        if value
    ).lower()

    mappings = (
        (("orthodont",), "Orthodontics"),
        (("operative", "conservative dentistry", "restorative dentistry"), "Operative Dentistry"),
        (("endodont",), "Endodontics"),
        (("oral surgery", "maxillofacial", "omfs"), "Oral & Maxillofacial Surgery"),
        (("prosthodont", "prostho"), "Prosthodontics"),
        (("periodont", "perio"), "Periodontology"),
        (("oral pathology", "oral path"), "Oral Pathology"),
        (("community dentistry", "dental public health"), "Community Dentistry"),
        (("pediatric", "paediatric", "pedodont"), "Pediatric Dentistry"),
        (("oral medicine",), "Oral Medicine"),
        (("general medicine",), "General Medicine"),
        (("general surgery",), "General Surgery"),
    )

    for cues, canonical in mappings:
        if any(cue in text for cue in cues):
            return canonical

    return ""


def _fallback_past_paper_subject(
    filename: str,
    title: str,
) -> str:
    canonical = canonical_bds_subject(
        filename,
        title,
    )

    if canonical:
        return canonical

    value = f"{filename} {title}".lower()

    mapping = (
        (("ortho", "orthodont"), "Orthodontics"),
        (("operative", "conservative"), "Operative Dentistry"),
        (("endodont", "endo "), "Endodontics"),
        (("oral surgery", "omfs", "maxillofacial"), "Oral & Maxillofacial Surgery"),
        (("prostho", "prosthodont"), "Prosthodontics"),
        (("periodont",), "Periodontology"),
        (("oral path", "oral pathology"), "Oral Pathology"),
        (("community", "public health"), "Community Dentistry"),
        (("medicine",), "General Medicine"),
        (("surgery",), "General Surgery"),
    )

    for cues, subject in mapping:
        if any(cue in value for cue in cues):
            return subject

    return "BDS Past Paper"


def looks_like_past_paper_pages(
    pages: List[Dict[str, Any]],
) -> bool:
    """Recognize exam/MCQ pages from already-OCRed RAG text."""
    text = "\n".join(
        normalize_text(page.get("text", ""))
        for page in pages
        if normalize_text(page.get("text", ""))
    )

    if len(text) < 120:
        return False

    lower = text.lower()

    exam_cues = (
        "multiple choice",
        "multiple-choice",
        "mcq",
        "mcqs",
        "question paper",
        "examination",
        "exam",
        "annual",
        "professional",
        "best answer",
        "choose the correct",
        "select the correct",
        "which of the following",
        "marks",
        "time allowed",
        "roll no",
        "roll number",
    )

    cue_hits = sum(
        1
        for cue in exam_cues
        if cue in lower
    )

    option_hits = len(
        re.findall(
            r"(?:^|\s)(?:\(?[a-eA-E]\)?[\)\].:\-]|"
            r"\(?[1-5]\)?[\)\].:\-])\s*\S+",
            text,
        )
    )

    numbered_hits = len(
        re.findall(
            r"(?:^|\n|\s)\d{1,3}[\)\].:\-]\s*\S+",
            text,
        )
    )

    question_hits = (
        text.count("?")
        + len(
            re.findall(
                r"\b(?:which|what|when|where|why|how)\b",
                lower,
            )
        )
    )

    return bool(
        (option_hits >= 8 and numbered_hits >= 3)
        or (option_hits >= 6 and cue_hits >= 2)
        or (
            cue_hits >= 3
            and numbered_hits >= 4
            and question_hits >= 2
        )
    )


def infer_existing_past_paper_metadata(
    document: Dict[str, Any],
    pages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    filename = normalize_text(
        document.get("filename", "")
    )
    title = normalize_text(
        document.get("title", "")
    )

    sample_parts = []
    sample_length = 0

    for page in pages[:20]:
        text = normalize_text(
            page.get("text", "")
        )

        if not text:
            continue

        block = (
            f"--- PDF PAGE {int(page.get('page', 0))} ---\n"
            f"{text}"
        )

        if sample_length + len(block) > 12000:
            remaining = max(
                0,
                12000 - sample_length,
            )

            if remaining:
                sample_parts.append(
                    block[:remaining]
                )
            break

        sample_parts.append(block)
        sample_length += len(block)

    forced_past_paper = (
        str(
            document.get("category", "")
        ).strip().lower()
        in {
            "past papers",
            "past paper",
        }
    )

    prompt = f"""
Determine whether this already-indexed Dentora document is a BDS past
examination paper or past-paper compilation and identify its metadata.

FILENAME
--------
{filename}

STORED TITLE
------------
{title}

STORED CATEGORY
---------------
{document.get("category", "")}

DOCUMENT SAMPLE
---------------
{chr(10).join(sample_parts)}

Return ONLY one JSON object:
{{
  "is_past_paper": true,
  "subject": "canonical BDS subject name",
  "year": "year if evident, otherwise Unknown",
  "title": "concise paper title"
}}

RULES
-----
1. A real past paper contains examination questions, MCQs, EMQs, essays,
   short-answer questions, OSCE stations, or an answer key from an exam.
2. Do not classify ordinary textbook prose as a past paper.
3. Infer the subject from the document itself when possible.
4. Do not invent a year. Use "Unknown" if no year is supported.
5. Return JSON only.
"""

    generated = generate_with_fallback(prompt)

    fallback_subject = _fallback_past_paper_subject(
        filename,
        title,
    )

    result = {
        "is_past_paper": forced_past_paper,
        "subject": fallback_subject,
        "year": "Unknown",
        "title": (
            title
            or os.path.splitext(filename)[0]
            or "Past Paper"
        ),
    }

    if generated.get("provider") == "Error":
        return result

    try:
        parsed = _json_payload(
            generated.get("response", ""),
            "object",
        )
    except Exception as exc:
        print(
            "Past-paper metadata inference warning:",
            exc,
        )
        return result

    if not isinstance(parsed, dict):
        return result

    result["is_past_paper"] = (
        forced_past_paper
        or bool(
            parsed.get(
                "is_past_paper",
                False,
            )
        )
    )

    parsed_subject = normalize_text(
        parsed.get("subject", "")
    )[:160]

    subject = (
        canonical_bds_subject(
            parsed_subject,
            filename,
            title,
        )
        or parsed_subject
    )[:160]

    year = normalize_text(
        parsed.get("year", "")
    )[:40]

    inferred_title = normalize_text(
        parsed.get("title", "")
    )[:300]

    if subject:
        result["subject"] = subject
    if year:
        result["year"] = year
    if inferred_title:
        result["title"] = inferred_title

    return result


def cross_check_past_paper_question(
    question: Dict[str, Any],
) -> Dict[str, Any]:
    options = question.get("options") or {}

    if not isinstance(options, dict) or len(options) < 2:
        return {
            "status": "not_auto_gradable",
            "verified_answer": "",
            "confidence": "",
            "rationale": (
                "This item does not contain enough structured answer options "
                "for automatic marking."
            ),
            "sources": [],
        }

    option_text = "\n".join(
        f"{key}. {value}"
        for key, value in options.items()
    )

    query = (
        f"{question.get('stem', '')}\n"
        f"{option_text}"
    )

    sources = rag_store.search(
        query,
        categories=["Books"],
        top_k=4,
        exclude_assessment=True,
    )

    if not sources:
        fallback_prompt = f"""
You are reviewing a BDS past-paper MCQ for practice.

QUESTION
--------
{question.get('stem', '')}

OPTIONS
-------
{option_text}

Choose the single best answer using standard undergraduate dental knowledge.
Return ONLY one JSON object:
{{
  "verified_answer": "one option label or empty string",
  "confidence": "high/moderate/low",
  "rationale": "one short explanation"
}}

If the OCR question/options are too damaged or genuinely ambiguous, return an
empty verified_answer. Do not invent a missing option.
"""

        generated = generate_with_fallback(
            fallback_prompt
        )

        if generated.get("provider") == "Error":
            return {
                "status": "insufficient",
                "verified_answer": "",
                "confidence": "low",
                "rationale": (
                    "No usable answer could be resolved automatically."
                ),
                "sources": [],
            }

        try:
            parsed = _json_payload(
                generated.get("response", ""),
                "object",
            )
        except Exception:
            parsed = {}

        fallback_answer = str(
            parsed.get(
                "verified_answer",
                "",
            )
            or ""
        ).strip().upper()

        valid_labels = {
            str(key).strip().upper()
            for key in options.keys()
        }

        if fallback_answer not in valid_labels:
            fallback_answer = ""

        return {
            "status": (
                "ai_reviewed"
                if fallback_answer
                else "insufficient"
            ),
            "verified_answer": fallback_answer,
            "confidence": str(
                parsed.get(
                    "confidence",
                    "low",
                )
                or "low"
            ).strip().lower(),
            "rationale": normalize_text(
                parsed.get(
                    "rationale",
                    "",
                )
            )[:1800],
            "sources": [],
        }

    context = build_context(sources)
    provided = str(
        question.get("provided_answer", "")
        or ""
    ).strip().upper()

    prompt = f"""
You are independently checking a BDS past-paper MCQ against retrieved
explanatory textbook evidence.

QUESTION
--------
{question.get('stem', '')}

OPTIONS
-------
{option_text}

PAST-PAPER KEY IF PRESENT
-------------------------
{provided or "No answer key was visible in the paper."}

RETRIEVED TEXTBOOK EVIDENCE
---------------------------
{context}

Return ONLY one JSON object:
{{
  "verified_answer": "A/B/C/D/etc or empty string",
  "confidence": "high/moderate/low",
  "rationale": "brief evidence-based explanation"
}}

STRICT RULES
------------
1. Determine the answer from explanatory textbook evidence, not from the
   past-paper key.
2. Review questions, MCQ distractors, and answer options inside a textbook
   excerpt are not evidence unless explanatory prose establishes them.
3. If the retrieved evidence does not clearly support one option, return an
   empty verified_answer and low confidence.
4. Never invent a page, source, option, or answer.
5. Keep the rationale concise and do not claim more than the excerpts support.
"""
    generated = generate_with_fallback(prompt)

    if generated.get("provider") == "Error":
        raise RuntimeError("No AI provider was available for cross-checking.")

    parsed = _json_payload(
        generated.get("response", ""),
        "object",
    )

    verified = str(
        parsed.get("verified_answer", "")
        or ""
    ).strip().upper()

    valid_labels = {
        str(key).strip().upper()
        for key in options.keys()
    }

    if verified not in valid_labels:
        verified = ""

    if not verified:
        fallback_prompt = f"""
Review this BDS MCQ and select the single best option.

QUESTION
--------
{question.get('stem', '')}

OPTIONS
-------
{option_text}

The retrieved textbook context below was relevant but not decisive:
{context}

Return ONLY JSON:
{{
  "verified_answer": "one option label or empty string",
  "confidence": "high/moderate/low",
  "rationale": "one short explanation"
}}

Use standard undergraduate dental knowledge together with the context. If the
OCR is too damaged or the item is genuinely ambiguous, return an empty answer.
"""
        fallback_generated = generate_with_fallback(
            fallback_prompt
        )

        try:
            fallback_parsed = (
                _json_payload(
                    fallback_generated.get(
                        "response",
                        "",
                    ),
                    "object",
                )
                if fallback_generated.get(
                    "provider"
                ) != "Error"
                else {}
            )
        except Exception:
            fallback_parsed = {}

        candidate = str(
            fallback_parsed.get(
                "verified_answer",
                "",
            )
            or ""
        ).strip().upper()

        if candidate in valid_labels:
            verified = candidate
            parsed = fallback_parsed
            status = "ai_reviewed"
        else:
            status = "insufficient"
    elif provided and provided in valid_labels and provided != verified:
        status = "conflict"
    else:
        status = "verified"

    return {
        "status": status,
        "verified_answer": verified,
        "confidence": str(
            parsed.get("confidence", "low")
            or "low"
        ).strip().lower(),
        "rationale": normalize_text(
            parsed.get("rationale", "")
        )[:1800],
        "sources": [
            {
                "filename": source.get("filename", ""),
                "title": source.get("title", ""),
                "page": int(source.get("page", 0) or 0),
                "category": source.get("category", ""),
            }
            for source in sources
        ],
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/")
def root():
    return {
        "message": "Dentora backend is running.",
        "version": "2.4.1-auto-detect-scanned-papers",
    }


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "Dentora API",
        "rag_configured": rag_store.configured,
        "voice_configured": bool(os.getenv("GROQ_API_KEY")),
        "test_engine_configured": past_paper_store.configured,
        "beta_access_required": True,
        "beta_access_configured": bool(DENTORA_BETA_ACCESS_CODE),
        "owner_access_configured": bool(DENTORA_OWNER_ACCESS_CODE),
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
    role = require_beta_access(x_dentora_beta_key)
    return {"ok": True, "access": role}


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
# VOICE TRANSCRIPTION
# =========================================================

@app.post("/transcribe")
async def transcribe_voice(
    request: Request,
    file: UploadFile = File(...),
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    require_beta_access(x_dentora_beta_key)

    enforce_rate_limit(
        request,
        "voice-transcribe",
        VOICE_RATE_LIMIT,
        VOICE_RATE_WINDOW_SECONDS,
    )

    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="Voice transcription is not configured on the server.",
        )

    filename = (file.filename or "voice.webm")[:180]
    content_type = (file.content_type or "application/octet-stream").lower()
    allowed_extensions = (
        ".flac", ".mp3", ".mp4", ".mpeg", ".mpga",
        ".m4a", ".ogg", ".wav", ".webm",
    )

    if (
        not content_type.startswith("audio/")
        and not filename.lower().endswith(allowed_extensions)
    ):
        raise HTTPException(
            status_code=400,
            detail="Please send a supported audio recording.",
        )

    max_bytes = MAX_VOICE_AUDIO_MB * 1024 * 1024
    contents = await file.read(max_bytes + 1)

    if not contents:
        raise HTTPException(
            status_code=400,
            detail="The audio recording was empty.",
        )

    if len(contents) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Voice recordings are limited to {MAX_VOICE_AUDIO_MB} MB. "
                "Please record a shorter question."
            ),
        )

    try:
        transcription = groq_client.audio.transcriptions.create(
            file=(filename, contents, content_type),
            model="whisper-large-v3-turbo",
            response_format="json",
            temperature=0.0,
            prompt=(
                "BDS dental education and viva terminology. Preserve dental, "
                "medical, pharmacology, anatomy, orthodontic, endodontic, "
                "oral-surgery and operative-dentistry terms and abbreviations."
            ),
        )

        transcript = str(getattr(transcription, "text", "") or "").strip()

        if not transcript:
            raise HTTPException(
                status_code=422,
                detail=(
                    "Dentora could not detect clear speech. "
                    "Please try again closer to the microphone."
                ),
            )

        return {
            "success": True,
            "transcript": transcript,
            "provider": "Groq Whisper",
            "model": "whisper-large-v3-turbo",
        }

    except HTTPException:
        raise
    except Exception as exc:
        print("Voice transcription error:", exc)
        raise HTTPException(
            status_code=502,
            detail=(
                "Voice transcription is temporarily unavailable. "
                "Please type your question or try again."
            ),
        ) from exc


# =========================================================
# PAST PAPER ENGINE / TEST MODE
# =========================================================

@app.post("/past-papers/ingest-pdf")
async def ingest_past_paper(
    request: Request,
    file: UploadFile = File(...),
    subject: str = Form(""),
    year: str = Form(""),
    title: str = Form(""),
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    require_owner_access(x_dentora_beta_key)
    enforce_rate_limit(
        request,
        "past-paper-ingest",
        5,
        60 * 60,
    )

    if not past_paper_store.configured:
        raise HTTPException(
            status_code=503,
            detail="Persistent question-bank storage is not configured.",
        )

    filename = file.filename or "past-paper.pdf"

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF past papers are supported.",
        )

    contents = await file.read(
        MAX_TEMP_PDF_MB * 1024 * 1024 + 1
    )

    if len(contents) > MAX_TEMP_PDF_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Past-paper ingestion accepts PDFs up to "
                f"{MAX_TEMP_PDF_MB} MB."
            ),
        )

    try:
        extraction = extract_pdf_pages(
            contents,
            allow_ocr=True,
            web_mode=True,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail="Dentora could not read this past-paper PDF.",
        ) from exc

    if extraction["page_count"] > 150:
        raise HTTPException(
            status_code=413,
            detail=(
                "Past-paper structuring currently accepts up to 150 pages "
                "per PDF. Split very large compilations into smaller files."
            ),
        )

    resolved_subject = normalize_text(subject)[:160]
    resolved_year = normalize_text(year)[:40]
    resolved_title = (
        normalize_text(title)
        or os.path.splitext(filename)[0]
    )[:300]

    if not resolved_subject:
        raise HTTPException(
            status_code=400,
            detail="Enter the subject before structuring the paper.",
        )

    try:
        questions = parse_past_paper_questions(
            pages=extraction["pages"],
            subject=resolved_subject,
        )

        questions = canonicalize_past_paper_topics(
            questions,
            resolved_subject,
        )

        if not questions:
            raise HTTPException(
                status_code=422,
                detail=(
                    "No clear examination questions could be structured "
                    "from this PDF."
                ),
            )

        paper_id = document_id(contents)

        result = past_paper_store.index_paper(
            paper_id=paper_id,
            title=resolved_title,
            subject=resolved_subject,
            year=resolved_year or "Unknown",
            filename=filename,
            questions=questions,
            replace_existing=True,
        )

        result["ocr_used"] = extraction["ocr_used"]
        result["unresolved_scanned_pages"] = (
            extraction["unresolved_scanned_pages"]
        )
        return result

    except HTTPException:
        raise
    except Exception as exc:
        print("Past-paper ingestion error:", exc)
        raise HTTPException(
            status_code=500,
            detail="Dentora could not structure this past paper.",
        ) from exc


@app.get("/past-papers/rag-documents")
def list_existing_rag_documents(
    request: Request,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    """Show the owner the documents already stored in the RAG library."""
    require_owner_access(x_dentora_beta_key)
    enforce_rate_limit(
        request,
        "past-paper-rag-documents",
        30,
        10 * 60,
    )

    imported_ids = {
        str(paper.get("paper_id", "")).strip()
        for paper in past_paper_store.list_papers()
    }

    documents = []

    for item in rag_store.list_documents():
        doc_id = str(item.get("doc_id", "")).strip()
        if not doc_id:
            continue

        filename = normalize_text(item.get("filename", ""))
        title = normalize_text(item.get("title", ""))
        category = normalize_text(item.get("category", "Other"))

        cue_text = f"{filename} {title} {category}".lower()
        suggested = (
            category.lower() in {"past papers", "past paper"}
            or bool(
                re.search(
                    r"(past\s*paper|question\s*paper|annual\s*exam|"
                    r"professional\s*exam|supply\s*exam|orthodont|"
                    r"operative)",
                    cue_text,
                    re.IGNORECASE,
                )
            )
        )

        documents.append({
            "doc_id": doc_id,
            "filename": filename,
            "title": title or filename,
            "category": category,
            "pages": int(item.get("pages", 0) or 0),
            "chunks": int(item.get("chunks", 0) or 0),
            "already_imported": doc_id in imported_ids,
            "suggested_past_paper": suggested,
        })

    documents.sort(
        key=lambda item: (
            not item["suggested_past_paper"],
            item["already_imported"],
            item["category"].lower(),
            item["title"].lower(),
        )
    )

    return {
        "count": len(documents),
        "documents": documents,
    }


@app.post("/past-papers/import-rag/{doc_id}")
def import_selected_rag_past_paper(
    doc_id: str,
    payload: ExistingPaperImportRequest,
    request: Request,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    """Import one exact already-indexed RAG document into Test Mode."""
    require_owner_access(x_dentora_beta_key)
    enforce_rate_limit(
        request,
        "past-paper-import-rag",
        20,
        60 * 60,
    )

    documents = rag_store.list_documents()
    document = next(
        (
            item
            for item in documents
            if str(item.get("doc_id", "")).strip() == doc_id
        ),
        None,
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="That RAG document was not found.",
        )

    pages = rag_store.document_pages(doc_id)

    if not pages:
        raise HTTPException(
            status_code=422,
            detail=(
                "Dentora found the RAG document but could not reconstruct "
                "readable page text from its stored chunks."
            ),
        )

    inferred = infer_existing_past_paper_metadata(
        document,
        pages,
    )

    subject = normalize_text(payload.subject or "")[:160]
    if not subject:
        subject = normalize_text(
            inferred.get("subject", "")
        )[:160]
    if not subject:
        subject = _fallback_past_paper_subject(
            document.get("filename", ""),
            document.get("title", ""),
        )

    year = normalize_text(payload.year or "")[:40]
    if not year:
        year = normalize_text(
            inferred.get("year", "")
        )[:40]
    year = year or "Unknown"

    title = normalize_text(payload.title or "")[:300]
    if not title:
        title = normalize_text(
            inferred.get("title", "")
        )[:300]
    if not title:
        title = (
            normalize_text(document.get("title", ""))
            or normalize_text(document.get("filename", ""))
            or "Past Paper"
        )[:300]

    try:
        questions = parse_past_paper_questions(
            pages=pages,
            subject=subject,
        )

        questions = canonicalize_past_paper_topics(
            questions,
            subject,
        )

        if not questions:
            raise HTTPException(
                status_code=422,
                detail=(
                    "The selected RAG document was found, but Dentora could "
                    "not extract clear examination questions from its stored text."
                ),
            )

        result = past_paper_store.index_paper(
            paper_id=doc_id,
            title=title,
            subject=subject,
            year=year,
            filename=document.get("filename", ""),
            questions=questions,
            replace_existing=True,
        )

        result.update({
            "imported": True,
            "source": "selected_rag_document",
            "original_category": document.get("category", ""),
        })

        return result

    except HTTPException:
        raise
    except Exception as exc:
        print("Selected RAG paper import error:", doc_id, exc)
        raise HTTPException(
            status_code=500,
            detail=(
                "Dentora found the document, but structuring it into "
                "Test Mode failed."
            ),
        ) from exc


@app.get("/past-papers/rag-sources")
def list_rag_sources_for_test_mode(
    request: Request,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    require_owner_access(x_dentora_beta_key)
    enforce_rate_limit(
        request,
        "past-paper-rag-sources",
        30,
        10 * 60,
    )

    documents = rag_store.list_documents()

    def likely(document: Dict[str, Any]) -> bool:
        category = str(
            document.get("category", "")
        ).strip().lower()

        value = (
            f"{document.get('filename', '')} "
            f"{document.get('title', '')}"
        ).lower()

        if category in {
            "past papers",
            "past paper",
        }:
            return True

        return bool(
            re.search(
                r"(past\s*paper|question\s*paper|annual\s*(exam|examination)|"
                r"supply\s*(exam|examination)|professional\s*(exam|examination))",
                value,
                re.IGNORECASE,
            )
        )

    return {
        "count": len(documents),
        "documents": [
            {
                "doc_id": str(document.get("doc_id", "")),
                "filename": document.get("filename", ""),
                "title": document.get("title", ""),
                "category": document.get("category", ""),
                "pages": int(document.get("pages", 0) or 0),
                "chunks": int(document.get("chunks", 0) or 0),
                "likely_past_paper": likely(document),
                "already_in_test_bank": any(
                    str(paper.get("paper_id", ""))
                    == str(document.get("doc_id", ""))
                    for paper in past_paper_store.list_papers()
                ),
            }
            for document in documents
        ],
    }


def structure_existing_rag_paper(
    document: Dict[str, Any],
    *,
    subject_override: str = "",
    year_override: str = "",
    title_override: str = "",
) -> Dict[str, Any]:
    doc_id = str(
        document.get("doc_id", "")
    ).strip()

    if not doc_id:
        raise ValueError(
            "The indexed document has no document ID."
        )

    pages = rag_store.document_pages(
        doc_id
    )

    if not pages:
        raise ValueError(
            "Dentora could not reconstruct readable page text from this indexed PDF."
        )

    metadata = infer_existing_past_paper_metadata(
        document,
        pages,
    )

    subject = (
        canonical_bds_subject(
            subject_override,
            metadata.get("subject", ""),
            document.get("filename", ""),
            document.get("title", ""),
        )
        or normalize_text(subject_override)
        or normalize_text(
            metadata.get("subject", "")
        )
        or _fallback_past_paper_subject(
            document.get("filename", ""),
            document.get("title", ""),
        )
    )[:160]

    year = (
        normalize_text(year_override)
        or normalize_text(
            metadata.get("year", "")
        )
        or "Unknown"
    )[:40]

    resolved_title = (
        normalize_text(title_override)
        or normalize_text(
            metadata.get("title", "")
        )
        or normalize_text(
            document.get("title", "")
        )
        or normalize_text(
            document.get("filename", "")
        )
        or "Past Paper"
    )[:300]

    questions = parse_past_paper_questions(
        pages=pages,
        subject=subject,
    )

    questions = canonicalize_past_paper_topics(
        questions,
        subject,
    )

    if not questions:
        raise ValueError(
            "No clear examination questions could be structured from this indexed PDF."
        )

    result = past_paper_store.index_paper(
        paper_id=doc_id,
        title=resolved_title,
        subject=subject,
        year=year,
        filename=document.get("filename", ""),
        questions=questions,
        replace_existing=True,
    )

    result.update({
        "imported": True,
        "source": "existing_rag_library",
        "original_category": document.get(
            "category",
            "",
        ),
    })

    return result


@app.post("/past-papers/import-source/{doc_id}")
def import_specific_rag_past_paper(
    doc_id: str,
    payload: ImportRagPastPaperRequest,
    request: Request,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    require_owner_access(
        x_dentora_beta_key
    )

    enforce_rate_limit(
        request,
        "past-paper-import-source",
        20,
        60 * 60,
    )

    document = next(
        (
            item
            for item in rag_store.list_documents()
            if str(item.get("doc_id", ""))
            == str(doc_id)
        ),
        None,
    )

    if document is None:
        raise HTTPException(
            status_code=404,
            detail="That RAG document was not found.",
        )

    try:
        return structure_existing_rag_paper(
            document,
            subject_override=(
                payload.subject or ""
            ),
            year_override=(
                payload.year or ""
            ),
            title_override=(
                payload.title or ""
            ),
        )
    except Exception as exc:
        print(
            "Specific RAG past-paper import error:",
            doc_id,
            exc,
        )
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@app.post("/past-papers/import-existing")
def import_existing_past_paper(
    request: Request,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    """Migrate one already-indexed RAG past paper into Test Mode.

    This prevents owners from having to upload the same PDF twice. Each call
    imports at most one paper so Render requests remain bounded; the frontend
    repeats the call until no eligible legacy papers remain.
    """
    require_owner_access(
        x_dentora_beta_key
    )

    enforce_rate_limit(
        request,
        "past-paper-import-existing",
        20,
        60 * 60,
    )

    if not (
        rag_store.configured
        and past_paper_store.configured
    ):
        raise HTTPException(
            status_code=503,
            detail=(
                "Persistent RAG/question-bank storage is not configured."
            ),
        )

    existing_ids = {
        str(
            paper.get(
                "paper_id",
                "",
            )
        )
        for paper in past_paper_store.list_papers()
    }

    documents = rag_store.list_documents()

    strong_candidates = []
    fallback_candidates = []
    content_candidates = []
    unknown_documents = []

    cue_pattern = re.compile(
        r"(past\s*paper|pastpaper|question\s*paper|annual\s*(exam|examination)|"
        r"supply\s*(exam|examination)|professional\s*(exam|examination)|"
        r"university\s*(exam|examination)|previous\s*year|previous\s*paper|"
        r"ortho.*(?:mcq|paper|exam)|operative.*(?:mcq|paper|exam)|"
        r"(?:mcq|paper|exam).*ortho|(?:mcq|paper|exam).*operative)",
        re.IGNORECASE,
    )

    for document in documents:
        doc_id = str(
            document.get(
                "doc_id",
                "",
            )
        ).strip()

        if (
            not doc_id
            or doc_id in existing_ids
        ):
            continue

        raw_category = str(
            document.get(
                "category",
                "",
            )
        ).strip()

        category_words = set(
            re.findall(
                r"[a-z]+",
                raw_category.lower(),
            )
        )

        name_text = (
            f"{document.get('filename', '')} "
            f"{document.get('title', '')}"
        )

        if (
            "past" in category_words
            and "paper" in category_words
        ):
            strong_candidates.append(
                document
            )
        elif cue_pattern.search(name_text):
            fallback_candidates.append(
                document
            )
        else:
            unknown_documents.append(
                document
            )

    # Older RAG uploads may have been saved under Books/Other. Inspect their
    # already-OCRed chunks so scanned exam papers are still discovered.
    if not (
        strong_candidates
        or fallback_candidates
    ):
        for document in unknown_documents:
            doc_id = str(
                document.get(
                    "doc_id",
                    "",
                )
            ).strip()

            try:
                sample_pages = (
                    rag_store.document_sample_pages(
                        doc_id,
                        max_pages=12,
                        max_chunks=48,
                    )
                )

                if looks_like_past_paper_pages(
                    sample_pages
                ):
                    content_candidates.append(
                        document
                    )
                    continue

                page_count = int(
                    document.get(
                        "pages",
                        0,
                    )
                    or 0
                )

                if (
                    sample_pages
                    and 0 < page_count <= 220
                ):
                    inferred = (
                        infer_existing_past_paper_metadata(
                            document,
                            sample_pages,
                        )
                    )

                    if inferred.get(
                        "is_past_paper"
                    ):
                        content_candidates.append(
                            document
                        )

            except Exception as exc:
                print(
                    "Past-paper content detection warning:",
                    doc_id,
                    exc,
                )

    candidates = (
        strong_candidates
        + fallback_candidates
        + content_candidates
    )

    if not candidates:
        return {
            "success": True,
            "imported": False,
            "message": (
                "No unimported past-paper documents were found "
                "in the existing RAG library."
            ),
        }

    for document in candidates:
        doc_id = str(
            document.get(
                "doc_id",
                "",
            )
        ).strip()

        try:
            result = structure_existing_rag_paper(
                document
            )

            return result

        except Exception as exc:
            print(
                "Existing past-paper import warning:",
                doc_id,
                exc,
            )
            continue

    return {
        "success": True,
        "imported": False,
        "message": (
            "Past-paper candidates were found, but Dentora could not "
            "structure readable questions from them."
        ),
    }


@app.post("/past-papers/{paper_id}/verify")
def verify_past_paper(
    paper_id: str,
    request: Request,
    batch_size: int = 4,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    require_owner_access(x_dentora_beta_key)
    enforce_rate_limit(
        request,
        "past-paper-verify",
        60,
        60 * 60,
    )

    batch_size = max(1, min(int(batch_size), 6))
    pending = past_paper_store.pending_questions(
        paper_id,
        limit=batch_size,
    )

    processed = 0

    for question in pending:
        try:
            result = cross_check_past_paper_question(
                question
            )

            past_paper_store.update_verification(
                question["id"],
                status=result["status"],
                verified_answer=result["verified_answer"],
                confidence=result["confidence"],
                rationale=result["rationale"],
                sources=result["sources"],
            )
            processed += 1

        except Exception as exc:
            print(
                "Past-paper verification error:",
                question.get("id"),
                exc,
            )

    counts = past_paper_store.refresh_manifest_counts(
        paper_id
    )

    remaining = len(
        past_paper_store.pending_questions(
            paper_id,
            limit=12,
        )
    )

    return {
        "success": True,
        "paper_id": paper_id,
        "processed": processed,
        **counts,
        "pending_count": remaining,
        "complete": remaining == 0,
    }


@app.get("/test/catalog")
def test_catalog(
    request: Request,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    require_beta_access(x_dentora_beta_key)
    enforce_rate_limit(
        request,
        "test-catalog",
        60,
        10 * 60,
    )

    if not past_paper_store.configured:
        return {
            "configured": False,
            "paper_count": 0,
            "question_count": 0,
            "eligible_test_questions": 0,
            "subjects": [],
            "years": [],
            "topics": [],
            "papers": [],
        }

    catalog = past_paper_store.catalog()

    try:
        rag_documents = rag_store.list_documents()
        rag_past_paper_count = 0

        for document in rag_documents:
            category_words = set(
                re.findall(
                    r"[a-z]+",
                    str(
                        document.get(
                            "category",
                            "",
                        )
                    ).lower(),
                )
            )

            if (
                "past" in category_words
                and "paper" in category_words
            ):
                rag_past_paper_count += 1

        catalog["rag_past_paper_count"] = max(
            rag_past_paper_count,
            int(
                catalog.get(
                    "paper_count",
                    0,
                )
                or 0
            ),
        )
        catalog["rag_sync_complete"] = (
            catalog.get("paper_count", 0)
            >= rag_past_paper_count
        )
    except Exception as exc:
        print(
            "Test catalog RAG sync status warning:",
            exc,
        )
        catalog["rag_past_paper_count"] = 0
        catalog["rag_sync_complete"] = False

    return catalog


@app.post("/test/start")
def start_test(
    config: TestStartRequest,
    request: Request,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    require_beta_access(x_dentora_beta_key)
    enforce_rate_limit(
        request,
        "test-start",
        30,
        10 * 60,
    )

    questions = past_paper_store.sample_test(
        count=config.count,
        subject=config.subject,
        year=config.year,
        topic=config.topic,
        weak_topics=config.weak_topics,
        repeated_only=config.repeated_only,
    )

    return {
        "count": len(questions),
        "questions": questions,
        "answers_hidden": True,
    }


@app.post("/test/grade")
def grade_test(
    payload: TestGradeRequest,
    request: Request,
    x_dentora_beta_key: Optional[str] = Header(
        default=None,
        alias="X-Dentora-Beta-Key",
    ),
):
    require_beta_access(x_dentora_beta_key)
    enforce_rate_limit(
        request,
        "test-grade",
        30,
        10 * 60,
    )

    return past_paper_store.grade(
        payload.responses
    )


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
