from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from pypdf import PdfReader
from pdf2image import convert_from_bytes
from google import genai
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
import pytesseract
import os
import ollama


# =========================
# ENVIRONMENT
# =========================

load_dotenv()


# =========================
# APP
# =========================

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================
# AI CLIENTS
# =========================

gemini_client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY")
)

groq_client = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

qwen_client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)


# =========================
# CHAT REQUEST
# =========================

class ChatRequest(BaseModel):
    message: str
    mode: str


# =========================
# PDF MEMORY
# =========================

uploaded_pdf_text = ""
uploaded_pdf_name = ""
uploaded_pdf_chunks = []


def create_pdf_chunks(text, chunk_size=3000, overlap=300):

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = min(
            start + chunk_size,
            text_length
        )

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


def find_relevant_pdf_chunks(
    query,
    chunks,
    max_chunks=5
):

    if not chunks:
        return []

    query_words = {
        word.lower().strip(
            ".,!?;:()[]{}"
        )
        for word in query.split()
        if len(word) > 2
    }

    scored_chunks = []

    for chunk in chunks:

        chunk_lower = chunk.lower()

        score = 0

        for word in query_words:

            if word in chunk_lower:
                score += 1

        scored_chunks.append(
            (score, chunk)
        )

    scored_chunks.sort(
        key=lambda item: item[0],
        reverse=True
    )

    relevant = [
        chunk
        for score, chunk in scored_chunks[:max_chunks]
        if score > 0
    ]

    return relevant


# =========================
# ROOT
# =========================

@app.get("/")
def root():

    return {
        "message": "Dentora backend is running."
    }


# =========================
# PDF UPLOAD
# =========================

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...)
):

    if not file.filename.lower().endswith(".pdf"):

        return {
            "success": False,
            "message": "Only PDF files are supported."
        }

    try:

        contents = await file.read()

        temp_path = Path(
            "temp_uploaded.pdf"
        )

        temp_path.write_bytes(
            contents
        )

        reader = PdfReader(
            str(temp_path)
        )

        text = ""

        for page in reader.pages:

            page_text = (
                page.extract_text()
                or ""
            )

            text += (
                page_text
                + "\n"
            )

        # OCR fallback for scanned PDFs
        if len(text.strip()) < 100:

            images = convert_from_bytes(
                contents,
                dpi=200
            )

            text = ""

            for image in images:

                page_text = (
                    pytesseract.image_to_string(
                        image
                    )
                )

                text += (
                    page_text
                    + "\n"
                )

        temp_path.unlink(
            missing_ok=True
        )

        global uploaded_pdf_text
        global uploaded_pdf_name
        global uploaded_pdf_chunks

        uploaded_pdf_text = text

        uploaded_pdf_name = (
            file.filename
        )

        uploaded_pdf_chunks = (
            create_pdf_chunks(text)
        )

        print(
            f"PDF uploaded: {file.filename}"
        )

        print(
            f"Pages: {len(reader.pages)}"
        )

        print(
            f"Characters: {len(text)}"
        )

        print(
            f"Chunks: {len(uploaded_pdf_chunks)}"
        )

        return {

            "success": True,

            "filename":
                file.filename,

            "pages":
                len(reader.pages),

            "text":
                text

        }

    except Exception as e:

        print(
            "PDF upload error:",
            e
        )

        return {

            "success": False,

            "message":
                str(e)

        }


# =========================
# CHAT
# =========================

@app.post("/chat")
def chat(
    request: ChatRequest
):

    # Student's actual question
    message = request.message


    # =========================
    # PDF RETRIEVAL
    # =========================

    if uploaded_pdf_chunks:

        relevant_chunks = (
            find_relevant_pdf_chunks(
                message,
                uploaded_pdf_chunks
            )
        )

        if relevant_chunks:

            relevant_pdf = (
                "\n\n".join(
                    relevant_chunks
                )
            )

        else:

            relevant_pdf = (
                "No directly relevant "
                "PDF material was found "
                "for this question."
            )

    else:

        relevant_pdf = (
            "No PDF has been uploaded."
        )


    # =========================
    # PROMPT
    # =========================

    prompt = f"""
You are Dentora by Ehsan — a dedicated BDS-level dental education tutor.

STUDENT:
The student is a final-year BDS student preparing for university examinations,
viva examinations, OSCEs, clinical discussions, and dental academic work.

CURRENT MODE:
{request.mode}

STUDENT'S QUESTION:
{message}

UPLOADED PDF MATERIAL:
---------------------
{relevant_pdf}

PDF INSTRUCTIONS:
----------------
If PDF material is available:

- Use it as the primary source for questions related to the uploaded material.
- Base explanations on the supplied PDF content.
- Prioritize the PDF's terminology, classifications, explanations, and sequence.
- You may add relevant BDS-level background knowledge when useful.
- Do not invent information that is not supported by the PDF or established medical/dental knowledge.
- If the PDF does not contain enough information to answer the question, say so briefly and then provide relevant background knowledge.

GENERAL INSTRUCTIONS:
---------------------
- Answer at final-year BDS level.
- Be accurate and clinically relevant.
- Explain concepts clearly.
- For examinations, emphasize high-yield points.
- For viva questions, give concise viva-ready answers.
- For OSCE questions, focus on identification, clinical findings, steps, interpretation, and key points.
- Use tables when useful.
- Use bullet points for examination-friendly information.
- Do not unnecessarily repeat the student's question.

Answer the student's question now.
"""


    # =========================
    # GEMINI
    # =========================

    try:

        response = (
            gemini_client.models.generate_content(
                model="gemini-3.5-flash",
                contents=prompt
            )
        )

        return {

            "response":
                response.text,

            "provider":
                "Gemini"

        }

    except Exception as gemini_error:

        print(
            "Gemini error:",
            gemini_error
        )

        error_text = str(
            gemini_error
        )

        if not any(
            code in error_text
            for code in [
                "429",
                "503",
                "RESOURCE_EXHAUSTED",
                "UNAVAILABLE"
            ]
        ):

            print(
                "Gemini error is not "
                "eligible for fallback."
            )

            return {

                "response":
                    "Dentora couldn't connect to the AI service. Please try again.",

                "provider":
                    "Error"

            }


    # =========================
    # GROQ FALLBACK
    # =========================

    try:

        response = (
            groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        )

        return {

            "response":
                response.choices[0].message.content,

            "provider":
                "Groq"

        }

    except Exception as groq_error:

        print(
            "Groq error:",
            groq_error
        )


    # =========================
    # QWEN API FALLBACK
    # =========================

    try:

        response = (
            qwen_client.chat.completions.create(
                model="qwen-plus",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
        )

        return {

            "response":
                response.choices[0].message.content,

            "provider":
                "Qwen API"

        }

    except Exception as qwen_error:

        print(
            "Qwen API error:",
            qwen_error
        )


    # =========================
    # LOCAL OLLAMA FALLBACK
    # =========================

    try:

        response = ollama.chat(

            model="qwen3:8b",

            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]

        )

        return {

            "response":
                response["message"]["content"],

            "provider":
                "Local Qwen"

        }

    except Exception as ollama_error:

        print(
            "Ollama error:",
            ollama_error
        )

        return {

            "response":
                "Dentora couldn't connect to the AI service. Please try again.",

            "provider":
                "Error"

        }