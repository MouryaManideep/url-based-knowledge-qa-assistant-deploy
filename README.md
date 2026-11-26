# 📚 Knowledge QA — ask your sources

A lightweight, production-minded RAG assistant. Paste web links or upload PDFs, then ask questions and get **grounded answers with numbered citations** you can check. Built with Streamlit and Google Gemini.

**Highlights**

- **Hybrid retrieval**: dense semantic search (local `bge-small-en-v1.5` embeddings) fused with BM25 keyword search using Reciprocal Rank Fusion, so exact names, codes and numbers aren't missed. Optional MMR for diverse passages.
- **Conversation-aware**: follow-up questions ("what about its cost?") are searched in the context of the previous question.
- **Streaming answers** with inline `[n]` citations; each source card shows the passage, section/page, match score and a link.
- **Strict or hybrid mode**: answer only from your sources, or allow clearly labelled general knowledge.
- **Summaries**: overview, key points, takeaways and suggested follow-up questions for all sources or one.
- **Source management**: web pages (HTML, plain text, or PDF links), PDF/.txt/.md uploads, duplicate detection, per-source removal, and an indexing progress bar for large files.
- **OCR for scans and photos**: scanned PDFs with no text layer and photos of documents (PNG, JPG, WebP) are read by Gemini, keeping page numbers for citations. Needs an API key; limited to 30 pages and 18 MB per file.
- **Export** a chat, with its cited sources, to Markdown.
- **Resilient model calls**: every request has a timeout and the whole retry chain a time budget, so nothing hangs. Transient errors (503/500/timeouts) are retried with jitter, then the app falls back through a configurable list of models; a model that just reported overload is skipped for 60 s. The UI shows what's happening and which model answered.

## Safe for many visitors

- **Per-session isolation**: each browser session has its own in-memory knowledge base and chat. Nothing is written to disk, and visitors never see each other's documents.
- **Per-session API keys**: a key pasted in the sidebar stays in that visitor's session. It is never written to the process environment.
- **Resource limits** per session (25 sources, 4,000 passages, 25 MB files, 500 PDF pages) protect a shared host.
- **SSRF protection** for URL fetching: public IPs only, every redirect re-checked, TLS verified, 5 MB cap.
- **Prompt-injection hardening**: retrieved text is fenced and marked untrusted, and images are stripped from model output so a poisoned page can't leak chat data through image URLs.

## Lite by design

No PyTorch, LangChain or FAISS. Embeddings run on ONNX Runtime via FastEmbed (about 70 MB model, CPU), and search is exact cosine plus BM25 in NumPy, which is sub-millisecond at per-session scale. The install is a fraction of the previous stack, which matters on Streamlit Community Cloud's 1 GB memory limit.

```text
app.py                 Streamlit UI
config.py              Settings and limits (env vars or Streamlit secrets)
loaders/               Safe web fetching, PDF text extraction (pypdf)
preprocessing/         HTML cleaning, recursive chunking
rag/                   Embeddings, BM25, hybrid index, prompts, Gemini client, QA
services/              Ingestion pipeline, summarizer
ui/                    Styles
tests/                 Unit tests and an app smoke test
```

## Run locally

Python 3.10–3.12:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env    # optionally set GEMINI_API_KEY
streamlit run app.py
```

Get a free Gemini key at <https://aistudio.google.com/apikey>. The embedding model downloads on first use.

## Deploy on Streamlit Community Cloud

1. Push this repo and create an app with `app.py` as the entry point. Choose Python 3.12 under *Advanced settings*.
2. Optional, under **Settings → Secrets**:

   ```toml
   GEMINI_API_KEY = "your-key"        # omit to make each visitor bring their own key
   GEMINI_MODEL = "gemini-3.8-flash"  # any Gemini model your key can use
   GEMINI_FALLBACK_MODELS = "gemini-3.6-flash,gemini-3.5-flash-lite"  # tried if the primary is overloaded
   ```

   If you set `GEMINI_API_KEY`, every visitor uses your key and quota. For a public demo, leaving it unset is usually the better choice.

## Tests

```bash
python -m unittest discover -s tests -t . -v
```

The smoke test runs the real app with Streamlit's `AppTest` when Streamlit is installed.

## Limitations

- Knowledge bases are session-only and disappear on refresh. This is a deliberate privacy choice.
- OCR sends the whole scanned file to Gemini and is limited to 30 pages per file. Pages that need JavaScript or a login may not load.
- URL checks don't pin DNS to the connection, so DNS rebinding can't be fully ruled out. If you expose this widely, add network-level egress controls.
- Questions and retrieved passages are sent to Google Gemini. Review Google's data terms before using confidential documents.
- Grounded prompting reduces unsupported answers but can't eliminate them, so verify important answers against the sources.
