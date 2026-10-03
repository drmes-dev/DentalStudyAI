"""Bounded chat context and encrypted, immutable MCQ state.

Keys stay on the server; the browser persists only an authenticated opaque token.
This lets grading survive reloads and deploys without another model call.
"""
import base64
import hashlib
import json
import re

from cryptography.fernet import Fernet, InvalidToken


def history_context(history):
    return json.dumps([
        {"role": turn.role, "content": turn.content[:6000]}
        for turn in history[-12:]
    ], ensure_ascii=False)


def contextual_query(message, history):
    """Resolve short follow-ups without embedding 'B' or 'next' on its own."""
    short = len(message.split()) <= 6 or re.search(
        r"\b(this|that|it|these|those|next|again|why)\b", message, re.I)
    if not short or not history:
        return message
    previous = next((t.content for t in reversed(history) if t.role == "user"
                     and len(t.content.split()) > 6), "")
    last_answer = next((t.content for t in reversed(history) if t.role == "assistant"), "")
    return f"{previous[:1500]}\n{last_answer[:2000]}\nFollow-up: {message}"


def next_question(message):
    return bool(re.fullmatch(r"\s*(?:give me |ask me )?(?:the |a |one )?"
                             r"(?:next|another|more)(?:\s+(?:mcq|question))?[.!?]*\s*",
                             message, re.I))


def question_request(message):
    return next_question(message) or bool(re.search(
        r"\b(?:give|ask|create|generate|make|quiz|test)\b.*\b(?:mcqs?|questions?|me)\b",
        message, re.I))


def answer_label(message):
    match = re.fullmatch(r"\s*(?:(?:my )?answer(?: is)?[: ]*|option\s+|i (?:choose|think)\s+)?"
                         r"([A-Ea-e])(?:[.)\-:]\s*[^\n]{0,200})?[.!?]?\s*", message)
    return match.group(1).upper() if match else None


class MCQTokens:
    def __init__(self, secret):
        key = hashlib.sha256(("dentora-mcq-v1:" + secret).encode()).digest()
        self.cipher = Fernet(base64.urlsafe_b64encode(key))

    def seal(self, state, session_id, chat_id):
        payload = {"version": 1, "session": session_id, "chat": chat_id, "state": state}
        return self.cipher.encrypt(json.dumps(payload).encode()).decode()

    def open(self, token, session_id, chat_id):
        if not token:
            return None
        try:
            payload = json.loads(self.cipher.decrypt(token.encode(), ttl=7 * 86400))
            if payload.get("version") != 1 or payload.get("session") != session_id or payload.get("chat") != chat_id:
                return None
            return payload["state"]
        except (InvalidToken, ValueError, KeyError, TypeError):
            return None


def validate_question(candidate, sources):
    """Reject malformed options and fabricated supporting quotations."""
    stem = candidate.get("stem")
    options = candidate.get("options")
    key = candidate.get("correct_answer")
    explanation = candidate.get("explanation")
    if not isinstance(stem, str) or not 15 <= len(stem) <= 1500:
        raise ValueError("Invalid MCQ stem")
    if not isinstance(options, dict) or list(options) not in [list("ABCD"), list("ABCDE")]:
        raise ValueError("MCQ needs ordered A-D or A-E options")
    if any(not isinstance(v, str) or not v.strip() or len(v) > 500 for v in options.values()):
        raise ValueError("Invalid option text")
    if len({v.casefold().strip() for v in options.values()}) != len(options) or key not in options:
        raise ValueError("Invalid or duplicate options/key")
    if not isinstance(explanation, str) or not 20 <= len(explanation) <= 2500:
        raise ValueError("Missing explanation")
    support = candidate.get("support")
    validate_support(support, sources, explanation)
    return {"stem": stem.strip(), "options": options, "correct_answer": key,
            "explanation": explanation.strip(), "support": support}


def render_question(question):
    options = "\n\n".join(f"**{key}.** {value}" for key, value in question["options"].items())
    return f"**Question**\n\n{question['stem']}\n\n{options}\n\nChoose one option (A–{list(question['options'])[-1]})."


def render_grade(question, label):
    key = question["correct_answer"]
    outcome = "Correct." if label == key else f"You chose {label}. The correct answer is {key}."
    return (f"**{outcome}**\n\n**{key}. {question['options'][key]}**\n\n"
            f"{question['explanation']}\n\nSay **next question** to continue on the same subject.")


def validate_support(support, sources, explanation):
    if not isinstance(support, list) or not support:
        raise ValueError("Missing textbook support")
    for item in support:
        label, quote = item.get("label", ""), item.get("quote", "")
        if not re.fullmatch(r"S[1-9]\d*", label) or not isinstance(quote, str) or len(quote.strip()) < 25:
            raise ValueError("Invalid evidence label/quote")
        index = int(label[1:]) - 1
        if index >= len(sources) or sources[index].get("evidence_kind") == "assessment":
            raise ValueError("Question options are not evidence")
        normalize = lambda s: re.sub(r"\s+", " ", s).strip().casefold()
        if normalize(quote) not in normalize(sources[index].get("text", "")):
            raise ValueError("Invented supporting quote")
    # A source label in the explanation must belong to the supplied support.
    supported = {item["label"] for item in support}
    if any(label not in supported for label in re.findall(r"\[(S\d+)\]", explanation)):
        raise ValueError("Unsupported explanation citation")
