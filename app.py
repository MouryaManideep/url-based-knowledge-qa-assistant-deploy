"""Knowledge QA — grounded answers from your web pages and documents.

Every browser session gets its own in-memory knowledge base and its own API key slot,
so the app is safe to host for many visitors at once.
"""
from __future__ import annotations

import html
from typing import Any, Dict, List

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import (  # noqa: E402  (settings read .env, so load it first)
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_MIN_RELEVANCE,
    DEFAULT_TOP_K,
    GEMINI_MODEL,
    MAX_QUESTION_CHARS,
    SAMPLE_URL,
    get_setting,
)
from rag.embeddings import FastEmbedder  # noqa: E402
from rag.index import KnowledgeBase  # noqa: E402
from rag.llm import GeminiClient, LLMError  # noqa: E402
from rag.prompts import NOT_FOUND  # noqa: E402
from rag.qa import cited_numbers, hits_to_citations, is_not_found, retrieval_queries, sanitize_markdown, stream_answer  # noqa: E402
from rag.types import SourceError  # noqa: E402
from services.ingest import file_source_id, ingest_file, ingest_url, url_source_id  # noqa: E402
from services.summarizer import summarize  # noqa: E402
from ui.styles import CSS, WELCOME  # noqa: E402
from utils.helpers import export_chat_to_markdown, is_valid_url, md_code, md_escape, parse_url_list, safe_link  # noqa: E402
from utils.logger import setup_logger  # noqa: E402

logger = setup_logger("app")
SERVER_API_KEY = get_setting("GEMINI_API_KEY") or ""
STARTER_QUESTIONS = [
    "Summarize the main points.",
    "What are the most important facts and figures?",
    "What conclusions or recommendations are made?",
]
ICONS = {"web": "🌐", "pdf": "📄", "text": "📝", "image": "🖼️"}
UPLOAD_TYPES = ["pdf", "txt", "md", "png", "jpg", "jpeg", "webp"]

st.set_page_config(page_title="Knowledge QA · Ask your sources", page_icon="📚", layout="wide", initial_sidebar_state="expanded")
st.markdown(CSS, unsafe_allow_html=True)


# --------------------------------------------------------------------------- state
def init_state() -> None:
    defaults: Dict[str, Any] = {
        "kb": KnowledgeBase(),
        "messages": [],
        "summary": None,
        "user_api_key": "",
        "pending_question": None,
        "pending_urls": [],
        "view": "Chat",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_state()
kb: KnowledgeBase = st.session_state.kb


@st.cache_resource(show_spinner=False)
def get_embedder() -> FastEmbedder:
    """One shared, stateless model per server process; documents stay per session."""
    return FastEmbedder()


def load_embedder() -> FastEmbedder:
    try:
        with st.spinner("Loading the embedding model (first run downloads about 70 MB)…"):
            return get_embedder()
    except Exception as exc:
        logger.error("Embedding model failed to load (%s).", type(exc).__name__)
        raise SourceError("The embedding model could not be loaded. Check the server's internet access and retry.") from exc


def active_api_key() -> str:
    return (st.session_state.user_api_key or "").strip() or SERVER_API_KEY


# ----------------------------------------------------------------------- callbacks
def ask(question: str) -> None:
    st.session_state.pending_question = question
    st.session_state.view = "Chat"


def queue_sample() -> None:
    st.session_state.pending_urls = [SAMPLE_URL]


def remove_source(source_id: str) -> None:
    kb.remove(source_id)
    st.session_state.summary = None


def clear_everything() -> None:
    kb.clear()
    st.session_state.messages = []
    st.session_state.summary = None


def clear_chat() -> None:
    st.session_state.messages = []


# ----------------------------------------------------------------------- ingestion
def make_ocr():
    """OCR for scans and photos, done by Gemini; None without an API key (the file is then explained, not read)."""
    api_key = active_api_key()
    if not api_key:
        return None

    def ocr(data: bytes, mime_type: str) -> str:
        st.caption("No text layer found, so Gemini is reading it (OCR). This can take a minute…")
        return GeminiClient(api_key, GEMINI_MODEL, on_event=lambda message: st.caption(message)).transcribe(data, mime_type)

    return ocr


def run_ingest(urls: List[str], files: List[Any]) -> None:
    jobs = [("url", u) for u in urls] + [("file", f) for f in files]
    if not jobs:
        st.toast("Add a link or a file first.", icon="ℹ️")
        return
    size, overlap = st.session_state.chunk_size, st.session_state.chunk_overlap
    ocr = make_ocr()
    added, failed = 0, 0
    with st.status(f"Adding {len(jobs)} source{'s' if len(jobs) > 1 else ''}…", expanded=True) as status:
        embedder = None
        for kind, item in jobs:
            label = item if kind == "url" else item.name
            shown = md_code(label) if kind == "url" else f"**{md_escape(label)}**"
            try:
                if kind == "url":
                    if not is_valid_url(item):
                        raise SourceError("Not a valid http(s) link.")
                    if kb.has_source(url_source_id(item)):
                        raise SourceError("Already in your knowledge base.")
                    kb.check_capacity()
                    st.caption(f"Fetching {md_code(label)}…")
                    source, chunks = ingest_url(item, size, overlap, ocr=ocr)
                else:
                    data = item.getvalue()
                    if kb.has_source(file_source_id(data)):
                        raise SourceError("Already in your knowledge base.")
                    kb.check_capacity()
                    source, chunks = ingest_file(item.name, data, size, overlap, ocr=ocr)
                kb.check_capacity(len(chunks))
                embedder = embedder or load_embedder()
                bar = st.progress(0.0, text=f"Indexing {len(chunks)} passages…")
                kb.add(source, chunks, embedder,
                       progress=lambda done, total: bar.progress(done / total, text=f"Indexing passages {done:,}/{total:,}…"))
                bar.empty()
                added += 1
                st.markdown(f"✅ **{md_escape(source.title)}** · {len(chunks)} passage{'s' if len(chunks) != 1 else ''}")
            except (SourceError, LLMError) as exc:
                failed += 1
                st.markdown(f"⚠️ {shown} — {md_escape(exc)}")
            except Exception as exc:  # Never let one bad source crash the page.
                failed += 1
                logger.exception("Unexpected ingestion failure (%s).", type(exc).__name__)
                st.markdown(f"⚠️ {shown} — Unexpected error ({type(exc).__name__}).")
        if added and not failed:
            status.update(label=f"Added {added} source{'s' if added > 1 else ''}", state="complete", expanded=False)
        elif added:
            status.update(label=f"Added {added}, skipped {failed}", state="complete", expanded=True)
        else:
            status.update(label="No sources were added", state="error", expanded=True)
    if added:
        st.session_state.summary = None


# ------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown('<div class="brand">📚 Knowledge QA<small>Answers grounded in your sources</small></div>', unsafe_allow_html=True)

    st.markdown("#### Gemini API key")
    if SERVER_API_KEY:
        st.caption("✅ This app has a key configured.")
        with st.expander("Use my own key instead"):
            st.text_input("Your Gemini API key", type="password", key="user_api_key", label_visibility="collapsed", placeholder="Paste your key")
    else:
        st.text_input("Your Gemini API key", type="password", key="user_api_key", label_visibility="collapsed", placeholder="Paste your key")
        st.caption("Stored only for this browser session. [Get a free key ↗](https://aistudio.google.com/apikey)")

    st.markdown("#### Add sources")
    with st.form("add_sources", clear_on_submit=True, border=False):
        urls_text = st.text_area("Web pages", placeholder="https://example.com/article\nOne link per line", height=96)
        uploads = st.file_uploader("Files", type=UPLOAD_TYPES, accept_multiple_files=True,
                                   help="PDF, image (PNG, JPG, WebP), .txt or .md, up to 25 MB each. "
                                        "Scanned PDFs and photos are read with Gemini OCR, which needs an API key.")
        submitted = st.form_submit_button("Add to knowledge base", type="primary")

    with st.expander("Answer & retrieval settings"):
        st.toggle("Answer only from my sources", value=True, key="strict", help="Off: the assistant may add clearly labelled general knowledge.")
        st.slider("Passages per answer", 3, 12, DEFAULT_TOP_K, key="top_k")
        st.slider("Minimum relevance", 0.0, 0.8, DEFAULT_MIN_RELEVANCE, 0.05, key="min_relevance",
                  help="Passages with lower semantic similarity are ignored unless they match your keywords.")
        st.toggle("Diversify passages", value=False, key="diversify", help="Maximal marginal relevance: fewer near-duplicate passages.")
        st.select_slider("Chunk size (characters)", options=[500, 750, 1000, 1500, 2000], value=DEFAULT_CHUNK_SIZE, key="chunk_size")
        st.select_slider("Chunk overlap", options=[0, 100, 150, 200, 300], value=DEFAULT_CHUNK_OVERLAP, key="chunk_overlap")
        st.caption("Chunk settings apply to sources you add next.")

    pending_urls = st.session_state.pending_urls
    st.session_state.pending_urls = []
    if submitted or pending_urls:
        run_ingest(parse_url_list(urls_text) + pending_urls if submitted else pending_urls, list(uploads or []) if submitted else [])

    st.divider()
    st.caption("🔒 Your sources and chat live only in this browser session's memory and are never shared with other visitors. "
               "Question text and retrieved passages are sent to Google Gemini to write answers.")


# -------------------------------------------------------------------------- header
st.markdown(
    f"""<div class="hero"><div><h1>Ask your sources</h1>
<p>Grounded answers from web pages and documents, with citations you can check.</p></div>
<div class="stats"><span><b>{len(kb.sources)}</b> sources</span><span><b>{len(kb):,}</b> passages</span>
<span>{html.escape(GEMINI_MODEL)}</span></div></div>""",
    unsafe_allow_html=True,
)

view = st.radio(
    "View", ["Chat", "Summary", "Sources"], horizontal=True, key="view", label_visibility="collapsed",
    format_func=lambda v: {"Chat": "💬 Chat", "Summary": "📝 Summary", "Sources": f"📚 Sources ({len(kb.sources)})"}[v],
)


# ---------------------------------------------------------------------------- chat
def render_citations(message: Dict[str, Any]) -> None:
    citations = message.get("citations") or []
    if not citations:
        return
    cited = set(message.get("cited") or [])
    shown = [c for c in citations if c["n"] in cited] or citations
    with st.expander(f"Sources · {len(shown)}"):
        for cite in shown:
            with st.container(border=True):
                title = f"**[{cite['n']}] {md_escape(cite['title'])}**"
                url = cite.get("url")
                if url and is_valid_url(url):
                    title += f" · [Open ↗]({safe_link(url)})"
                st.markdown(title)
                meta = [md_escape(cite["location"])] if cite.get("location") else []
                meta.append(f"{cite['relevance']:.0%} semantic match")
                if cite.get("keyword"):
                    meta.append("keyword match")
                st.caption(" · ".join(meta))
                snippet = cite.get("snippet") or ""
                st.caption(md_escape(snippet[:320] + ("…" if len(snippet) > 320 else "")))


def render_model_note(message: Dict[str, Any]) -> None:
    model = message.get("model")
    if model and model != GEMINI_MODEL:
        st.caption(f"Answered by {md_code(model)} because {md_code(GEMINI_MODEL)} was busy.")


def answer(question: str) -> None:
    question = question.strip()[:MAX_QUESTION_CHARS]
    if not question:
        return
    messages: List[Dict[str, Any]] = st.session_state.messages
    history = [m for m in messages if not m.get("error")]
    with st.chat_message("user"):
        st.markdown(question)
    messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        reply: Dict[str, Any] = {"role": "assistant", "content": "", "citations": [], "cited": []}
        api_key = active_api_key()
        if not api_key:
            reply.update(content="Add a Gemini API key in the sidebar to get answers.", error=True)
            st.markdown(reply["content"])
            messages.append(reply)
            return
        try:
            with st.spinner("Searching your sources…"):
                hits = kb.search(
                    retrieval_queries(question, history), load_embedder(),
                    k=st.session_state.top_k, min_relevance=st.session_state.min_relevance, diversify=st.session_state.diversify,
                )
        except SourceError as exc:
            reply.update(content=f"⚠️ {exc}", error=True)
            st.markdown(reply["content"])
            messages.append(reply)
            return

        strict = st.session_state.strict
        if not hits and strict:
            reply["content"] = (f"{NOT_FOUND}\n\nTry rephrasing, lowering **Minimum relevance** in the settings, "
                                "or adding a source that covers this topic.")
            st.markdown(reply["content"])
            messages.append(reply)
            return

        placeholder = st.empty()
        placeholder.markdown("_Thinking…_")
        buffer = ""
        try:
            llm = GeminiClient(api_key, GEMINI_MODEL, on_event=lambda message: placeholder.markdown(f"_{message}_"))
            for delta in stream_answer(llm, question, hits, history, strict=strict):
                buffer += delta
                placeholder.markdown(sanitize_markdown(buffer) + " ▌")
            text = sanitize_markdown(buffer).strip() or "The model returned an empty response. Please try again."
            reply.update(content=text, citations=[] if is_not_found(text) else hits_to_citations(hits),
                         cited=sorted(cited_numbers(text, len(hits))), model=llm.last_model)
        except LLMError as exc:
            reply.update(content=f"⚠️ {exc}", error=True)
        placeholder.markdown(reply["content"])
        render_model_note(reply)
        render_citations(reply)
        messages.append(reply)


def render_chat() -> None:
    messages = st.session_state.messages
    if kb.is_empty and not messages:
        st.markdown(WELCOME, unsafe_allow_html=True)
        left, mid, right = st.columns([1, 1.2, 1])
        mid.button("✨ Try it with a sample page", on_click=queue_sample, type="primary")
        st.caption("The sample loads the Wikipedia article on retrieval-augmented generation.", help=SAMPLE_URL)

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_model_note(message)
                render_citations(message)

    if not messages and not kb.is_empty:
        st.markdown("##### Try asking")
        cols = st.columns(len(STARTER_QUESTIONS))
        for col, starter in zip(cols, STARTER_QUESTIONS):
            col.button(starter, on_click=ask, args=(starter,))

    typed = st.chat_input(
        "Ask a question about your sources…" if not kb.is_empty else "Add a source to start asking questions",
        disabled=kb.is_empty, max_chars=MAX_QUESTION_CHARS,
    )
    question = typed or st.session_state.pending_question
    st.session_state.pending_question = None
    if question and not kb.is_empty:
        answer(question)

    if st.session_state.messages:
        st.write("")
        c1, c2, _ = st.columns([1, 1, 3])
        c1.download_button("⬇️ Export chat", export_chat_to_markdown(st.session_state.messages),
                           file_name="chat.md", mime="text/markdown")
        c2.button("🗑️ Clear chat", on_click=clear_chat)


# ------------------------------------------------------------------------- summary
def render_summary() -> None:
    if kb.is_empty:
        st.info("Add a source to generate a summary.")
        return
    options = [None] + list(kb.sources)
    if st.session_state.get("summary_scope") not in options:
        st.session_state.summary_scope = None
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    scope = c1.selectbox("Summarize", options, key="summary_scope",
                         format_func=lambda s: "All sources" if s is None else kb.sources[s].title)
    if c2.button("Generate summary", type="primary"):
        api_key = active_api_key()
        if not api_key:
            st.warning("Add a Gemini API key in the sidebar first.")
        else:
            with st.status("Reading your sources…", expanded=True) as status:
                try:
                    llm = GeminiClient(api_key, GEMINI_MODEL, on_event=lambda message: st.caption(message))
                    result = summarize(llm, kb, scope)
                    st.session_state.summary = {"scope": scope, "model": llm.last_model, **result}
                    status.update(label="Summary ready", state="complete", expanded=False)
                except LLMError as exc:
                    status.update(label="The summary couldn't be generated", state="error", expanded=True)
                    st.error(str(exc))

    summary = st.session_state.summary
    if not summary or summary.get("scope") != scope:
        return
    with st.container(border=True):
        st.markdown("#### Overview")
        st.markdown(summary["overview"])
        st.markdown("#### Key points")
        st.markdown("\n".join(f"- {p}" for p in summary["key_points"]))
        if summary.get("takeaways"):
            st.markdown("#### Takeaways")
            st.markdown("\n".join(f"- {t}" for t in summary["takeaways"]))
        st.caption(summary["coverage"] + (f" Written by {summary['model']}." if summary.get("model") else ""))
    if summary.get("questions"):
        st.markdown("##### Explore further")
        for i, q in enumerate(summary["questions"]):
            st.button(f"💬 {q}", key=f"suggest-{i}", on_click=ask, args=(q,))


# ------------------------------------------------------------------------- sources
def render_sources() -> None:
    if kb.is_empty:
        st.info("No sources yet. Add web pages or files from the sidebar.")
        return
    for source in sorted(kb.sources.values(), key=lambda s: s.added_at):
        with st.container(border=True):
            c1, c2 = st.columns([5, 1], vertical_alignment="center")
            c1.markdown(f"{ICONS.get(source.kind, '📄')} **{md_escape(source.title)}**")
            origin = f"[{md_escape(source.origin)}]({safe_link(source.origin)})" if is_valid_url(source.origin) else md_escape(source.origin)
            c1.caption(f"{origin} · {source.chunk_count} passages · {source.char_count:,} characters")
            c2.button("Remove", key=f"rm-{source.id}", on_click=remove_source, args=(source.id,))
    st.caption(f"{len(kb.sources)} of {kb.max_sources} sources · {len(kb):,} of {kb.max_chunks:,} passages used this session")
    st.button("Remove all sources and chat", on_click=clear_everything)


{"Chat": render_chat, "Summary": render_summary, "Sources": render_sources}[view]()
