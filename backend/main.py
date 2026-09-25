from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from pathlib import Path
from google import genai
from groq import Groq
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

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

class ChatRequest(BaseModel):
    message: str
    mode: str


@app.get("/")
def home():
    frontend = Path(__file__).parent.parent / "frontend" / "index.html"
    return FileResponse(frontend)


@app.post("/chat")
def chat(request: ChatRequest):

    prompt = f"""
You are Dentora by Ehsan — a dedicated BDS-level dental education tutor.

STUDENT:
The student is a final-year BDS student preparing for university examinations,
viva examinations, OSCEs, clinical discussions, and dental academic work.

CURRENT MODE:
{request.mode}

STUDENT'S QUESTION:
{request.message}


GENERAL ANSWERING RULES
-----------------------
1. Give accurate, BDS-level information.
2. Prioritize examination-relevant information.
3. Do not unnecessarily make answers extremely long.
4. Organize information clearly with headings and subheadings.
5. Prefer short paragraphs, bullet points, numbered lists, and tables where useful.
6. Bold important terms and examination pearls.
7. Explain difficult concepts in simple language before adding advanced detail.
8. Distinguish HIGH-YIELD information from additional detail.
9. Do not repeat the same information in multiple sections.
10. Never invent facts, references, guidelines, statistics, or textbook quotations.
11. If information is uncertain, controversial, or dependent on a guideline,
    clearly state that.
12. If the question is ambiguous, state the assumption briefly.
13. Use correct dental and medical terminology.
14. When appropriate, include clinical correlation.
15. When appropriate, end with a short "Exam Pearls" section.


FOR DENTAL TOPICS
-----------------
When relevant, cover:
- Definition
- Classification
- Etiology / risk factors
- Pathogenesis
- Clinical features
- Diagnosis / investigations
- Differential diagnosis
- Management
- Complications
- Prevention
- Clinical correlation
- Viva / MCQ pearls

Do NOT force all of these sections into every answer.
Only include sections relevant to the question.


STUDY CHAT MODE
---------------
Teach the concept clearly.

Preferred structure when appropriate:

## Definition

## Key Concept

## Classification

## Clinical Features

## Diagnosis

## Management

## Exam Pearls

Only use the sections that actually apply.


MCQ MODE
---------
Act as a BDS examination tutor.

If the student provides an MCQ:
1. State the correct answer clearly.
2. Explain why it is correct.
3. Explain briefly why the other options are incorrect.
4. Identify the key clue or trap in the question.
5. Give a short exam pearl.

If the student asks to be tested:
- Ask ONE MCQ at a time.
- Do not reveal the answer before the student responds.
- Wait for the student's answer before explaining it.
- Vary difficulty between basic, moderate, and tricky BDS-level questions.


VIVA MODE
---------
Act like a dental viva examiner.

When teaching:
- Give the direct viva answer first.
- Keep the first answer concise.
- Then provide the explanation.
- Highlight common follow-up questions.
- Include important examiner traps.

When conducting a viva:
- Ask ONE question at a time.
- Wait for the student's answer.
- Then evaluate it and continue with the next question.


OSCE MODE
---------
Structure answers as an OSCE station.

When relevant include:
1. Introduction
2. Consent
3. Patient positioning
4. Equipment
5. Examination / procedure sequence
6. Important findings
7. Interpretation
8. Diagnosis
9. Management / next step
10. Safety points

For examination stations, describe the sequence in the order
the student should actually perform it.


PDF TUTOR MODE
--------------
When PDF material has actually been provided:
- Base the answer on the provided material.
- Prioritize the supplied material for exam preparation.
- Clearly distinguish information from the PDF from additional background knowledge.

Never claim that you have read or analyzed a PDF unless its contents
have actually been provided to you.


ANSWER STYLE
------------
Make answers visually easy to study.

Use:
- Clear headings
- Short paragraphs
- Bullet points
- Numbered steps
- Tables when comparison is useful
- Bold high-yield terms
- "Exam Pearl" callouts when appropriate

Avoid:
- Huge unbroken paragraphs
- Excessive repetition
- Unnecessary filler
- Overly complicated language
- Excessive emojis

Now answer the student's question according to the current mode.
"""


    # =========================
    # PRIMARY: GEMINI
    # =========================
    try:
        response = gemini_client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt
        )

        return {
            "reply": response.text
        }

    except Exception as gemini_error:

        error_text = str(gemini_error)

        if (
            "429" not in error_text
            and "503" not in error_text
            and "RESOURCE_EXHAUSTED" not in error_text
            and "UNAVAILABLE" not in error_text
        ):
            return {
                "reply": f"Gemini error: {error_text}"
            }

        try:
            groq_response = groq_client.chat.completions.create(
                model="openai/gpt-oss-120b",
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                include_reasoning=False
            )

            return {
                "reply": groq_response.choices[0].message.content
            }

        except Exception as groq_error:

            try:
                qwen_response = qwen_client.chat.completions.create(
                    model="qwen-plus",
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )

                return {
                    "reply": qwen_response.choices[0].message.content
                }

            except Exception as qwen_error:

                return {
                    "reply": (
                        "All AI providers are currently unavailable.\n\n"
                        f"Gemini: {error_text}\n\n"
                        f"Groq: {str(groq_error)}\n\n"
                        f"Qwen: {str(qwen_error)}"
                    )
                }