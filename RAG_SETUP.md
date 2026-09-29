# Dentora Persistent RAG — Zero-Cost Setup

Dentora now supports a persistent evidence-grounded RAG library.

## What is stored

The RAG system does **not need to permanently store the original PDF**.

For every PDF, Dentora stores:
- document name and category
- page number
- chunk number
- extracted page text
- semantic embedding
- optional permanent source URL

That design is intentional. Render's free filesystem is ephemeral, while Pinecone's Starter tier persists the searchable knowledge base.

## Zero-cost architecture

1. **Frontend:** GitHub Pages
2. **API:** existing Render free web service
3. **Vector database:** Pinecone Starter
4. **Embeddings:** Pinecone Inference, model `llama-text-embed-v2`
5. **Answer generation:** existing Gemini → Groq → Qwen → Ollama chain
6. **Large/scanned PDF ingestion:** your own Windows computer using `tools/index_library.py`

The original PDFs can later live in TeraBox/R2/another store. Dentora's RAG does not depend on that storage to answer questions.

## Why Pinecone

At the time this RAG version was built, Pinecone's Starter tier offers persistent vector storage and a monthly free allowance for hosted embedding inference. This prevents Render from needing a large embedding model in RAM.

Free tiers can change, so always check the current Pinecone limits before bulk-indexing a very large library.

## 1. Create Pinecone account

Create a free Pinecone account and create/copy an API key.

You do **not** need to manually create the Dentora index. The backend/indexer creates `dentora-knowledge` automatically if it does not exist.

## 2. Add Render environment variables

In Render → Dentora backend → Environment, add:

```
PINECONE_API_KEY=YOUR_KEY
PINECONE_INDEX_NAME=dentora-knowledge
PINECONE_CLOUD=aws
PINECONE_REGION=us-east-1
PINECONE_EMBED_MODEL=llama-text-embed-v2
PINECONE_EMBED_DIMENSION=1024
DENTORA_ADMIN_KEY=YOUR_PRIVATE_RANDOM_ADMIN_KEY
```

Never put either secret in `docs/index.html` or commit it to GitHub.

After saving variables, redeploy Render.

Then open:

```
https://dentalstudyai.onrender.com/rag/status
```

Expected:

```json
{
  "configured": true,
  "index_name": "dentora-knowledge",
  "embedding_model": "llama-text-embed-v2",
  "persistent": true
}
```

## 3. Organize your local library

Recommended folder structure:

```
DentoraLibrary/
├── Books/
├── Past Papers/
├── Viva/
├── OSCE/
└── Notes/
```

## 4. Index a normal PDF

From the repository root:

```bat
cd "C:\Users\IT MART\DentalStudyAI"
.venv\Scripts\activate
python tools\index_library.py "C:\path\to\book.pdf" --category "Books"
```

## 5. Index a whole library folder

```bat
python tools\index_library.py "D:\DentoraLibrary" --auto-category
```

The script:
1. opens one PDF at a time
2. extracts text page-by-page
3. OCRs only pages that need OCR
4. splits text into overlapping chunks
5. creates embeddings with Pinecone Inference
6. uploads the chunks to Pinecone
7. writes a persistent document manifest
8. skips exact duplicate files automatically

Because one file is handled at a time, you do not need to keep the entire library loaded in RAM.

## 6. Scanned PDFs

Your Windows setup already uses Tesseract and Poppler.

If they are on PATH:

```bat
python tools\index_library.py "D:\DentoraLibrary" --auto-category
```

If not:

```bat
python tools\index_library.py "D:\DentoraLibrary" --auto-category ^
  --tesseract-path "C:\Program Files\Tesseract-OCR\tesseract.exe" ^
  --poppler-path "C:\path\to\poppler\Library\bin"
```

## 7. Categories

Supported categories:

- Books
- Past Papers
- Viva
- OSCE
- Notes
- Other

These are stored as metadata and can later be used for category-specific retrieval.

## 8. Duplicate protection

A SHA-256 fingerprint is calculated for every PDF.

If the exact file is already indexed, the indexer skips it.

To intentionally replace it:

```bat
python tools\index_library.py "C:\book.pdf" --category "Books" --force
```

## 9. RAG answering behavior

For each question Dentora:

1. embeds the question
2. retrieves semantic candidates from Pinecone
3. boosts exact dental terminology/keyword overlap
4. limits duplicate chunks from the same page/document
5. gives the top evidence chunks to the LLM
6. requires `[S1]`, `[S2]` source labels
7. returns structured source metadata to the frontend
8. explicitly says when the uploaded library does not cover the requested point

The prompt forbids Dentora from pretending background knowledge came from the library.

## 10. Temporary PDF Tutor

The paperclip still works.

A temporary PDF:
- is separated by browser session ID
- is used first for PDF Tutor questions
- is **not** permanently stored on Render
- disappears when the free Render service restarts

Use the persistent indexer for books/notes you want to remain in the Dentora library.

## 11. API endpoints

Public/read-only:

```
GET  /rag/status
GET  /rag/library
POST /rag/search
POST /chat
```

Protected with header `X-Dentora-Admin-Key`:

```
POST   /rag/ingest-pdf
DELETE /rag/document/{doc_id}
```

This prevents friends using the public Dentora website from poisoning or deleting your permanent library.

## 12. Important capacity rule

The limiting resource is the persistent vector database/inference free tier, not the original PDF size.

A 500 MB scanned textbook can produce far less than 500 MB of searchable text. Therefore a large raw-PDF collection can sometimes fit into a much smaller semantic index.

However, "zero cost" does **not** mean unlimited. If your indexed chunks exceed Pinecone's free limits, do not enable billing automatically. Pause ingestion and either:
- wait for monthly inference allowance to reset,
- reduce/merge chunking,
- split the project,
- or move the vector layer to another free/local option.

## 13. Research-paper readiness

Each vector stores:
- document identity
- category
- page
- chunk index
- retrieval score

That gives Dentora an auditable retrieval trail suitable for later evaluation of:
- retrieval precision
- citation accuracy
- unsupported-claim rate
- hallucination rate
- RAG vs non-RAG performance
- category-specific performance
- exam/viva/OSCE usefulness

Do not delete this metadata if you intend to publish the Dentora study later.
