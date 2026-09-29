r"""
Dentora zero-cost local library indexer.

Use this for large or scanned PDFs so Render does not need to OCR
hundreds of pages inside one HTTP request.

Examples:
    python tools/index_library.py "C:\DentoraLibrary\Books\Davidson.pdf" --category "Books"
    python tools/index_library.py "C:\DentoraLibrary" --auto-category

Required in .env:
    PINECONE_API_KEY=...
"""

import argparse
import os
import sys
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from pypdf import PdfReader
from pdf2image import convert_from_path
import pytesseract


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"
sys.path.insert(0, str(BACKEND))

load_dotenv(ROOT / ".env")
load_dotenv(BACKEND / ".env")

from rag import document_id, normalize_text, rag_store, safe_category  # noqa: E402


CATEGORY_NAMES = {
    "books": "Books",
    "book": "Books",
    "past papers": "Past Papers",
    "past_papers": "Past Papers",
    "pastpapers": "Past Papers",
    "papers": "Past Papers",
    "viva": "Viva",
    "vivas": "Viva",
    "osce": "OSCE",
    "osces": "OSCE",
    "notes": "Notes",
    "note": "Notes",
}


def infer_category(path: Path) -> str:
    for part in reversed(path.parts):
        key = part.strip().lower()
        if key in CATEGORY_NAMES:
            return CATEGORY_NAMES[key]
    return "Other"


def configure_tesseract(explicit_path: str = ""):
    candidates = [
        explicit_path,
        os.getenv("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    ]

    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate

    return ""


def extract_pages(
    pdf_path: Path,
    use_ocr: bool,
    poppler_path: str,
) -> tuple[list[dict], list[int], int]:
    reader = PdfReader(str(pdf_path))
    pages = []
    needs_ocr = []

    for number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if len(text) < 40:
            needs_ocr.append(number)

        pages.append({
            "page": number,
            "text": text,
        })

    if use_ocr and needs_ocr:
        print(f"  OCR needed on {len(needs_ocr)} / {len(reader.pages)} pages")

        for counter, page_number in enumerate(needs_ocr, start=1):
            print(
                f"  OCR page {page_number}/{len(reader.pages)} "
                f"({counter}/{len(needs_ocr)})",
                end="\r",
                flush=True,
            )

            kwargs = {
                "dpi": 180,
                "first_page": page_number,
                "last_page": page_number,
                "fmt": "png",
                "thread_count": 1,
            }

            if poppler_path:
                kwargs["poppler_path"] = poppler_path

            images = convert_from_path(
                str(pdf_path),
                **kwargs,
            )

            if not images:
                continue

            text = normalize_text(
                pytesseract.image_to_string(images[0])
            )

            if text:
                pages[page_number - 1]["text"] = text

        print()

    unresolved = [
        page["page"]
        for page in pages
        if len(normalize_text(page.get("text", ""))) < 40
    ]

    return pages, unresolved, len(reader.pages)


def index_one(
    pdf_path: Path,
    category: str,
    title: str,
    source_url: str,
    force: bool,
    use_ocr: bool,
    poppler_path: str,
):
    print(f"\nIndexing: {pdf_path}")

    data = pdf_path.read_bytes()
    doc_id = document_id(data)

    if rag_store.manifest_exists(doc_id) and not force:
        print("  Already indexed — skipped.")
        return

    pages, unresolved, page_count = extract_pages(
        pdf_path,
        use_ocr=use_ocr,
        poppler_path=poppler_path,
    )

    readable_pages = page_count - len(unresolved)

    if readable_pages <= 0:
        print("  ERROR: no readable text was extracted.")
        return

    if unresolved:
        print(
            f"  Warning: {len(unresolved)} page(s) still have little/no text: "
            + ", ".join(str(value) for value in unresolved[:20])
            + ("..." if len(unresolved) > 20 else "")
        )

    result = rag_store.index_pages(
        doc_id=doc_id,
        filename=pdf_path.name,
        title=title or pdf_path.stem,
        category=safe_category(category),
        pages=pages,
        source_url=source_url,
        replace_existing=force,
    )

    print(
        "  DONE — "
        f"{result.get('pages', page_count)} pages, "
        f"{result.get('chunks', 0)} chunks, "
        f"category={result.get('category', category)}"
    )


def collect_pdfs(source: Path) -> list[Path]:
    if source.is_file():
        if source.suffix.lower() != ".pdf":
            raise SystemExit("The selected file is not a PDF.")
        return [source]

    if not source.is_dir():
        raise SystemExit("Source path does not exist.")

    return sorted(
        path
        for path in source.rglob("*.pdf")
        if path.is_file()
    )


def main():
    parser = argparse.ArgumentParser(
        description="OCR and index PDFs into Dentora's persistent Pinecone RAG library."
    )

    parser.add_argument("source", help="PDF file or folder containing PDFs.")
    parser.add_argument(
        "--category",
        default="Other",
        choices=["Books", "Past Papers", "Viva", "OSCE", "Notes", "Other"],
        help="Category for all selected PDFs.",
    )
    parser.add_argument(
        "--auto-category",
        action="store_true",
        help="Infer category from folder names such as Books, Past Papers, Viva, OSCE, Notes.",
    )
    parser.add_argument("--title", default="", help="Custom title (single PDF only).")
    parser.add_argument("--source-url", default="", help="Optional permanent URL to the original PDF.")
    parser.add_argument("--force", action="store_true", help="Replace a document that is already indexed.")
    parser.add_argument("--no-ocr", action="store_true", help="Do not OCR scanned pages.")
    parser.add_argument(
        "--tesseract-path",
        default="",
        help="Path to tesseract.exe if it is not on PATH.",
    )
    parser.add_argument(
        "--poppler-path",
        default=os.getenv("POPPLER_PATH", ""),
        help="Path to Poppler bin folder if pdftoppm is not on PATH.",
    )

    args = parser.parse_args()

    if not rag_store.configured:
        raise SystemExit(
            "PINECONE_API_KEY is missing. Put it in .env before running the indexer."
        )

    source = Path(args.source).expanduser().resolve()
    pdfs = collect_pdfs(source)

    if not pdfs:
        raise SystemExit("No PDFs found.")

    if len(pdfs) > 1 and args.title:
        raise SystemExit("--title can only be used when indexing one PDF.")

    if not args.no_ocr:
        selected = configure_tesseract(args.tesseract_path)
        if selected:
            print(f"Tesseract: {selected}")
        else:
            print(
                "Warning: Tesseract executable was not found. "
                "Text PDFs will still work, but scanned pages may fail."
            )

    print(f"Found {len(pdfs)} PDF(s).")
    print("Pinecone index:", os.getenv("PINECONE_INDEX_NAME", "dentora-knowledge"))

    failures = []

    for pdf_path in pdfs:
        try:
            category = infer_category(pdf_path) if args.auto_category else args.category

            index_one(
                pdf_path=pdf_path,
                category=category,
                title=args.title if len(pdfs) == 1 else "",
                source_url=args.source_url if len(pdfs) == 1 else "",
                force=args.force,
                use_ocr=not args.no_ocr,
                poppler_path=args.poppler_path,
            )

        except KeyboardInterrupt:
            print("\nStopped by user.")
            raise

        except Exception as exc:
            failures.append((pdf_path, str(exc)))
            print(f"  FAILED: {exc}")

    print("\nIndexing finished.")

    if failures:
        print(f"{len(failures)} file(s) failed:")
        for path, error in failures:
            print(f"  - {path.name}: {error}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
