import json
import os
import shutil
import tempfile
import uuid
from typing import List, Optional

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from embeddings.embedding_model import EMBEDDING_MODEL_NAME, get_embedding_model
from utils.logger import setup_logger

logger = setup_logger("faiss_store")
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_index")
STORE_PATH = os.path.join(INDEX_DIR, "store.json")
MANIFEST_PATH = os.path.join(INDEX_DIR, "manifest.json")


def _save_local(db: FAISS) -> None:
    """Write vectors and JSON metadata into a staged directory, then swap it in."""
    parent = os.path.dirname(INDEX_DIR)
    os.makedirs(parent, exist_ok=True)
    staging_dir = tempfile.mkdtemp(prefix=".faiss-staging-", dir=parent)
    backup_dir = f"{INDEX_DIR}.backup-{uuid.uuid4().hex}"
    old_index_moved = False
    try:
        faiss.write_index(db.index, os.path.join(staging_dir, "index.faiss"))
        documents = {}
        index_mapping = {}
        for position, document_id in db.index_to_docstore_id.items():
            document = db.docstore.search(document_id)
            if not isinstance(document, Document):
                raise ValueError("FAISS docstore contains an invalid record")
            documents[document_id] = {
                "page_content": document.page_content,
                "metadata": document.metadata,
            }
            index_mapping[str(int(position))] = document_id

        with open(os.path.join(staging_dir, "store.json"), "w", encoding="utf-8") as stream:
            json.dump({"documents": documents, "index_to_docstore_id": index_mapping}, stream, ensure_ascii=False)
        with open(os.path.join(staging_dir, "manifest.json"), "w", encoding="utf-8") as stream:
            json.dump({
                "format_version": 1,
                "embedding_model": EMBEDDING_MODEL_NAME,
                "dimension": int(db.index.d),
            }, stream)

        if os.path.exists(INDEX_DIR):
            os.replace(INDEX_DIR, backup_dir)
            old_index_moved = True
        try:
            os.replace(staging_dir, INDEX_DIR)
        except Exception:
            if old_index_moved and os.path.exists(backup_dir):
                os.replace(backup_dir, INDEX_DIR)
                old_index_moved = False
            raise
        if old_index_moved:
            shutil.rmtree(backup_dir, ignore_errors=True)
            old_index_moved = False
    finally:
        if os.path.exists(staging_dir):
            shutil.rmtree(staging_dir, ignore_errors=True)


def _load_safe(embeddings, model_name: str) -> FAISS:
    """Load only the non-executable JSON metadata format."""
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    if not (os.path.exists(index_path) and os.path.exists(STORE_PATH) and os.path.exists(MANIFEST_PATH)):
        if os.path.exists(os.path.join(INDEX_DIR, "index.pkl")):
            raise RuntimeError("The saved index uses legacy pickle metadata. Clear it and re-add your sources.")
        raise FileNotFoundError("The saved vector index is incomplete")

    with open(MANIFEST_PATH, encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("format_version") != 1 or manifest.get("embedding_model") != model_name:
        raise RuntimeError("The saved index is incompatible with the configured embedding model. Clear it and re-add sources.")

    index = faiss.read_index(index_path)
    if int(index.d) != int(manifest.get("dimension", -1)):
        raise RuntimeError("The saved vector index dimension does not match its manifest")
    with open(STORE_PATH, encoding="utf-8") as stream:
        payload = json.load(stream)

    documents = {
        document_id: Document(page_content=value["page_content"], metadata=value.get("metadata", {}))
        for document_id, value in payload["documents"].items()
    }
    index_mapping = {int(position): document_id for position, document_id in payload["index_to_docstore_id"].items()}
    if len(index_mapping) != int(index.ntotal) or any(document_id not in documents for document_id in index_mapping.values()):
        raise RuntimeError("The saved vector index and document metadata do not match")
    return FAISS(embeddings, index, InMemoryDocstore(documents), index_mapping)


def get_faiss_index(documents: Optional[List[Document]] = None, force_recreate: bool = False) -> Optional[FAISS]:
    """Load or update a local FAISS index without executing pickle metadata."""
    index_path = os.path.join(INDEX_DIR, "index.faiss")
    if not force_recreate and not documents and not os.path.exists(index_path):
        return None
    embeddings = get_embedding_model()
    if force_recreate:
        if not documents:
            clear_faiss_index()
            return None
        db = FAISS.from_documents(documents, embeddings)
        _save_local(db)
        return db

    if os.path.exists(index_path):
        try:
            db = _load_safe(embeddings, EMBEDDING_MODEL_NAME)
        except Exception as exc:
            logger.error("Could not load the saved index (%s).", type(exc).__name__)
            if documents:
                raise RuntimeError(str(exc)) from exc
            return None
        if documents:
            # Update a fresh in-memory copy, then atomically persist it. Never discard
            # old documents as a fallback when add/save fails.
            db.add_documents(documents)
            _save_local(db)
        return db

    if documents:
        db = FAISS.from_documents(documents, embeddings)
        _save_local(db)
        return db
    return None


def clear_faiss_index() -> None:
    """Delete the local vector index and its metadata directory."""
    if os.path.exists(INDEX_DIR):
        shutil.rmtree(INDEX_DIR)
