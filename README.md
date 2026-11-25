# URL-Based Knowledge QA Assistant

A Streamlit application that builds a local knowledge base from public web pages and uploaded text-based PDFs. It embeds document chunks locally with Sentence Transformers, searches them with FAISS, and uses Google Gemini to draft answers from retrieved passages. The app also offers source summaries and Markdown chat export.

## How it works

1. Add HTTP(S) URLs and/or upload PDF files in the Streamlit sidebar.
2. The loaders fetch webpage HTML or extract PDF text page by page.
3. Preprocessing cleans webpage content and splits text into overlapping chunks with source metadata.
4. A local Sentence Transformers model (`all-MiniLM-L6-v2`, normalized vectors) embeds chunks; FAISS stores the vectors and their document metadata under `faiss_index/`.
5. For a question, FAISS performs dense semantic retrieval. The app can return the closest semantic matches or use maximal marginal relevance (MMR) to diversify them. A configurable minimum cosine-similarity estimate filters weak matches; the estimate is derived from squared L2 distance for normalized vectors.
6. The app sends accepted passages, the question, and chat context to Gemini. Citations show the retrieval match score to help users inspect evidence quality.
7. Streamlit displays the answer and source details. Source metadata is rendered as plain text and only validated HTTP(S) sources become clickable links. Citations are useful references, but do not guarantee that every answer statement is fully supported.

## Project layout

```text
app.py                  Streamlit UI and orchestration
loaders/                Web and PDF content loading
preprocessing/          HTML cleanup and document chunking
embeddings/             Local sentence-transformer model
vectorstore/            FAISS index lifecycle
rag/                    Retrieval prompt and Gemini answer generation
services/               Summarization
utils/                  URL validation, hashing, export, logging
faiss_index/            Generated local data; ignored by Git
.env.example            Configuration template; contains no credentials
requirements.txt        Pinned direct Python dependencies
```

## Setup

Use Python 3.10–3.12 in a virtual environment:

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, then set `GEMINI_API_KEY` in `.env` for local development. Never commit `.env` or paste a key into source code, issues, or chat. On Streamlit Community Cloud, use the app's Secrets settings instead. If a key has ever been committed or shared, revoke it and create a replacement; deleting the file alone does not remove it from Git history.

Run the app:

```bash
streamlit run app.py
```

The embedding model is downloaded when first needed. Gemini access is needed for answers and summaries. The local FAISS index is generated as content is ingested and can be cleared from the sidebar. PDFs are limited to 25 MB, 500 pages, and 5 million extracted characters.

## Tests

Run the standard-library test suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

## Security and limitations

- URL ingestion accepts HTTP(S), validates DNS results and redirects, verifies TLS certificates, allows HTML only, and limits response size. Only ingest pages you are authorized to access. This is a basic server-side request forgery (SSRF) mitigation; DNS rebinding and hostile infrastructure cannot be ruled out in every deployment. Do not expose this app as a public multi-user service without a stronger egress/network policy.
- The FAISS index is local generated data and is excluded from Git. New indexes use FAISS vector files with JSON document metadata; old indexes that contain pickle metadata are not loaded. Clear an old index and re-add its sources to use the safer format.
- Embedding and retrieval are local. Prompts and retrieved content are sent to Google's Gemini API when answering or summarizing; review Google's applicable data terms before using confidential documents.
- Scanned/image-only PDFs are not OCR processed. Web pages that require login or client-side rendering may not load.
- Web URL checks reject non-public DNS addresses and validate redirects, but the loader does not pin a DNS resolution to the socket connection. Do not expose URL ingestion as a public server without network-level egress controls.
- Strict-source prompting reduces unsupported answers but cannot guarantee zero hallucinations. Verify important answers against the original source.
