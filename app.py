import os
# Disable Streamlit file watcher for smoother local development in this project.
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"

import streamlit as st
from dotenv import load_dotenv
from typing import List, Dict, Any

# File: app.py
# Purpose: Main entry point for the Streamlit web app.
# Flow: user provides URLs/PDFs -> content is processed -> embeddings are stored -> chat and summaries use the knowledge base.

# Load environment variables such as the Gemini API key from a local .env file.
load_dotenv()

# Setup page configuration for the web interface.
st.set_page_config(
    page_title="URL-Based RAG QA Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling
st.markdown("""
<style>
    /* Dark glassmorphic headers and backgrounds */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B 0%, #4D96FF 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1.5rem;
    }
    
    .stButton>button {
        border-radius: 8px;
        background: linear-gradient(135deg, #4D96FF 0%, #3B82F6 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: linear-gradient(135deg, #3B82F6 0%, #1D4ED8 100%);
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);
        transform: translateY(-1px);
    }
    
    .citation-card {
        padding: 10px;
        border-radius: 8px;
        background-color: #f0f4f8;
        border-left: 4px solid #4D96FF;
        margin-bottom: 8px;
        font-size: 0.85rem;
    }
    
    .dark .citation-card {
        background-color: #1e293b;
        border-left: 4px solid #3b82f6;
    }
</style>
""", unsafe_allow_html=True)

from loaders.web_loader import load_url_content
from loaders.pdf_loader import load_pdf_content
from preprocessing.cleaner import clean_html
from preprocessing.chunker import chunk_web_content, chunk_pdf_content
from vectorstore.faiss_store import INDEX_DIR, get_faiss_index, clear_faiss_index
from rag.qa_chain import generate_answer
from services.summarizer import get_summaries
from utils.helpers import is_valid_url, export_chat_to_markdown, content_fingerprint


def render_citations(citations):
    """Render untrusted source metadata as plain text and only link HTTP(S) URLs."""
    for cite in citations:
        title = str(cite.get("title") or "Untitled source")
        heading = str(cite.get("heading") or "Unlabelled section")
        page = cite.get("page")
        page_text = f", page {page}" if page else ""
        with st.container():
            st.text(f"{title} — {heading}{page_text}")
            score = cite.get("semantic_similarity")
            if isinstance(score, (int, float)):
                st.caption(f"Semantic match: {score:.0%}")
            source = str(cite.get("source") or "Unknown source")
            if is_valid_url(source):
                st.link_button("Open source", source)
            else:
                st.caption(f"Source: {source}")


# Initialize session state so information stays available across Streamlit reruns.
# This helps the app remember chat history, processed sources, and the FAISS database.
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_urls" not in st.session_state:
    st.session_state.processed_urls = set()
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}
if "db" not in st.session_state:
    # Try loading an existing FAISS index from disk when the app starts.
    st.session_state.db = get_faiss_index()

# Header: App overview
# Description: This app lets users add websites and PDFs, build a searchable knowledge base, and ask questions about the loaded content.

# Main page title shown at the top of the app.
st.title("URL-Based Intelligent QA Assistant")

# Header: Sidebar controls
# Description: Collects API keys, search settings, URLs, PDFs, and build actions.
with st.sidebar:
    st.header("⚙️ Configuration")
    if st.session_state.db is None and os.path.exists(os.path.join(INDEX_DIR, "index.pkl")) and not os.path.exists(os.path.join(INDEX_DIR, "store.json")):
        st.warning("This local index uses an older pickle format and was not loaded for safety. Clear it, then re-add your sources.")
    
    # API Key Input
    api_key_input = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="If you don't have it set in your .env file, enter it here."
    )
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input

    # Strict Source Mode Checkbox
    strict_mode = st.checkbox(
        "Strict Source Mode",
        value=True,
        help="When checked, the assistant only answers using the provided sources. When unchecked, it will fall back to general knowledge if the answer isn't in the sources."
    )
        
    st.markdown("---")
    
    # Dynamic settings for RAG and LLM
    st.subheader("🤖 Model & RAG Settings")
    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.1, step=0.05, help="Lower value is more deterministic, higher is more creative.")
    search_type = st.selectbox(
        "Semantic Search Strategy",
        options=["semantic_similarity", "mmr"],
        index=0,
        format_func=lambda x: "Semantic Similarity" if x == "semantic_similarity" else "Semantic MMR (diverse results)",
        help="Both options search embeddings, not exact keywords. MMR favors a diverse set of relevant passages.",
    )
    top_k = st.slider("Retrieval Chunk Count (K)", min_value=1, max_value=15, value=5, step=1)
    min_similarity = st.slider(
        "Minimum semantic similarity",
        min_value=0.0,
        max_value=0.8,
        value=0.20,
        step=0.05,
        help="Passages below this cosine-similarity estimate are ignored. Raise it to be stricter; lower it to retrieve broader matches.",
    )
    
    st.subheader("✂️ Chunking Parameters")
    chunk_size = st.slider("Chunk Size (characters)", min_value=200, max_value=2000, value=800, step=100)
    chunk_overlap = st.slider(
        "Chunk Overlap (characters)",
        min_value=0,
        max_value=min(500, chunk_size - 1),
        value=min(150, chunk_size - 1),
        step=50,
        help="Overlap preserves context between adjacent chunks and must remain smaller than the chunk size.",
    )

    st.markdown("---")
    
    # Input URLs
    st.subheader("🌐 Web Sources")
    urls_input = st.text_area(
        "Enter URLs (one per line)",
        placeholder="https://example.com\nhttps://another-page.com"
    )
    
    st.markdown("---")
    
    # Upload PDFs
    st.subheader("📄 PDF Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    # Build Knowledge Base Action
    build_db = st.button("Build Knowledge Base . . . . ")
    
    # Clear index action
    if st.button("Clear Database . . . . "):
        try:
            clear_faiss_index()
            st.session_state.db = None
            st.session_state.processed_urls.clear()
            st.session_state.processed_files.clear()
            st.session_state.chat_history.clear()
            st.success("Database cleared!")
            st.rerun()
        except OSError as exc:
            st.error(f"Could not completely clear the local database ({type(exc).__name__}).")

# Header: Build knowledge base
# Description: Process the entered URLs and uploaded PDFs into searchable chunks.
if build_db:
    # Split the multiline URL text into a clean list of links.
    urls = [u.strip() for u in urls_input.split("\n") if u.strip()]
    invalid_urls = [u for u in urls if not is_valid_url(u)]
    valid_urls = [u for u in urls if is_valid_url(u)]
    
    if invalid_urls:
        st.sidebar.warning(f"Skipped invalid URLs: {', '.join(invalid_urls)}")
        
    all_documents = []
    pending_urls = []
    pending_files = {}
    ingestion_errors = []
    
    # Process Web URLs
    if valid_urls:
        with st.spinner("Processing Web URLs..."):
            for url in valid_urls:
                if url in st.session_state.processed_urls:
                    continue
                try:
                    content = load_url_content(url)
                    if content:
                        sections = clean_html(content["html"])
                        chunks = chunk_web_content(sections, url=content["url"], title=content["title"], chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                        if chunks:
                            all_documents.extend(chunks)
                            pending_urls.append(url)
                        else:
                            ingestion_errors.append(f"No readable text found at {url}")
                    else:
                        ingestion_errors.append(f"Could not load {url}")
                except Exception as exc:
                    ingestion_errors.append(f"Could not process {url} ({type(exc).__name__})")
                    
    # Process Uploaded PDFs
    if uploaded_files:
        with st.spinner("Processing PDF Documents..."):
            for uploaded_file in uploaded_files:
                file_bytes = uploaded_file.getvalue()
                file_hash = content_fingerprint(file_bytes)
                if file_hash in st.session_state.processed_files:
                    continue
                # Load PDF from bytes
                try:
                    pages_data = load_pdf_content(file_bytes, uploaded_file.name)
                    chunks = chunk_pdf_content(pages_data, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                    if chunks:
                        all_documents.extend(chunks)
                        pending_files[file_hash] = uploaded_file.name
                    else:
                        ingestion_errors.append(f"No extractable text found in {uploaded_file.name}")
                except Exception as exc:
                    ingestion_errors.append(f"Could not process {uploaded_file.name} ({type(exc).__name__})")
                
    if all_documents:
        try:
            with st.spinner("Generating embeddings & updating FAISS index..."):
                db = get_faiss_index(all_documents)
            if db is None:
                raise RuntimeError("index update returned no database")
            st.session_state.db = db
            st.session_state.processed_urls.update(pending_urls)
            st.session_state.processed_files.update(pending_files)
            st.sidebar.success(f"Successfully processed {len(all_documents)} new chunks!")
            if ingestion_errors:
                st.sidebar.warning("Some sources were skipped. Check the source list and try those files/URLs again.")
            st.rerun()
        except Exception as exc:
            st.sidebar.error(f"The knowledge base was not updated ({type(exc).__name__}). Your sources remain eligible for retry.")
    else:
        if ingestion_errors:
            st.sidebar.warning("No new documents were indexed. One or more sources could not be read; correct them and retry.")
        else:
            st.sidebar.info("No new documents or URLs were added.")

# Header: Main content tabs
# Description: Organize the app into source viewing, summaries, and chat.
tab_chat, tab_summary, tab_sources = st.tabs(["💬 Chat Assistant", "📝 Summarization", "📚 Active Sources"])

# Header: Active sources view
with tab_sources:
    st.subheader("Currently Loaded Sources")
    
    if st.session_state.db:
        try:
            chunk_count = len(st.session_state.db.docstore._dict)
            st.info(f"📊 **Database Stats**: Currently storing **{chunk_count} chunks** in local FAISS vector store.")
        except Exception:
            st.info("📊 **Database Stats**: Chunks successfully stored in vector database.")
    else:
        st.info("📊 **Database Stats**: Vector database is currently empty.")
        
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Webpages:**")
        if st.session_state.processed_urls:
            for url in st.session_state.processed_urls:
                st.text(url)
                if is_valid_url(url):
                    st.link_button("Open webpage", url, key=f"source-{content_fingerprint(url.encode())[:12]}")
        else:
            st.info("No webpages loaded.")
    with col2:
        st.markdown("**PDF Documents:**")
        if st.session_state.processed_files:
            for filename in st.session_state.processed_files.values():
                st.text(f"PDF: {filename}")
        else:
            st.info("No PDF files uploaded.")

# Header: Summary view
with tab_summary:
    st.subheader("Source Material Summary")
    if not st.session_state.db:
        st.warning("Please build the knowledge base first to generate summaries.")
    else:
        if st.button("⚡ Generate Summaries"):
            with st.spinner("Analyzing sources and generating summaries..."):
                summaries = get_summaries(st.session_state.db, temperature=temperature)
            if summaries.get("error"):
                st.error(summaries["error"])
            else:
                if summaries.get("coverage"):
                    st.caption(summaries["coverage"])
                st.markdown("### 📊 Executive Summary")
                st.write(summaries["executive"])
                st.markdown("---")
                st.markdown("### 📌 Key Points")
                st.write(summaries["bullets"])
                st.markdown("---")
                st.markdown("### 💡 Takeaways")
                st.write(summaries["takeaways"])

# Header: Chat view
with tab_chat:
    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "citations" in message and message["citations"]:
                    with st.expander("🔍 Show Source Citations"):
                        render_citations(message["citations"])

    # Chat input
    if question := st.chat_input("Ask a question about the provided sources..."):
        if not os.getenv("GEMINI_API_KEY"):
            st.error("Please enter a Gemini API Key in the sidebar or check your .env configuration.")
        elif not st.session_state.db:
            st.warning("Please populate the knowledge base first by adding URLs or PDFs.")
        else:
            # Display user message
            with st.chat_message("user"):
                st.markdown(question)
            
            # Add to history
            st.session_state.chat_history.append({"role": "user", "content": question})
            
            # Generate response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                with st.spinner("Searching knowledge base and thinking..."):
                    try:
                        answer, retrieved_docs = generate_answer(
                            question=question,
                            db=st.session_state.db,
                            chat_history=st.session_state.chat_history[:-1], # Exclude current question
                            strict=strict_mode,
                            k=top_k,
                            temperature=temperature,
                            search_type=search_type,
                            min_similarity=min_similarity,
                        )
                    except Exception as exc:
                        answer, retrieved_docs = f"I couldn't complete that request ({type(exc).__name__}). Please try again.", []
                    
                    message_placeholder.markdown(answer)
                    
                    # Extract unique citations
                    citations = []
                    seen_citations = set()
                    for doc in retrieved_docs:
                        key = (doc.metadata.get("source"), doc.metadata.get("title"), doc.metadata.get("heading"), doc.metadata.get("page"))
                        if key not in seen_citations:
                            seen_citations.add(key)
                            citations.append({
                                    "title": doc.metadata.get("title"),
                                "heading": doc.metadata.get("heading"),
                                "page": doc.metadata.get("page"),
                                "source": doc.metadata.get("source"),
                                "semantic_similarity": doc.metadata.get("semantic_similarity"),
                            })
                            
                    # Show citations below answer
                    if citations:
                        with st.expander("🔍 Show Source Citations"):
                            render_citations(citations)
                                
            # Add assistant message to history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": answer,
                "citations": citations
            })

    # Export & download options
    if st.session_state.chat_history:
        st.markdown("---")
        md_export = export_chat_to_markdown(st.session_state.chat_history)
        st.download_button(
            label="📥 Export Chat History to Markdown",
            data=md_export,
            file_name="chat_history.md",
            mime="text/markdown"
        )
