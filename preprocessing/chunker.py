from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict, Any
from langchain_core.documents import Document

# File: preprocessing/chunker.py
# Purpose: Split cleaned text into smaller chunks that can be stored in the vector database.

def _make_text_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    if not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if not isinstance(chunk_overlap, int) or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be a non-negative integer smaller than chunk_size")
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, length_function=len)


def chunk_web_content(sections: List[Dict[str, Any]], url: str, title: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    """
    Chunks sections of web content, preserving source metadata (URL, title, heading).
    """
    text_splitter = _make_text_splitter(chunk_size, chunk_overlap)
    
    documents = []
    for sec in sections:
        text = str(sec.get("text", "")).strip()
        heading = str(sec.get("heading", "Introduction"))
        if not text:
            continue
        
        # Split text into chunks
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": url,
                    "title": title,
                    "heading": heading,
                    "type": "web",
                    "page": ""
                }
            )
            documents.append(doc)
            
    return documents

def chunk_pdf_content(pages_data: List[Dict[str, Any]], chunk_size: int = 800, chunk_overlap: int = 150) -> List[Document]:
    """
    Chunks pages of PDF content, preserving page numbers and source metadata.
    """
    text_splitter = _make_text_splitter(chunk_size, chunk_overlap)
    
    documents = []
    for page in pages_data:
        text = str(page.get("text", "")).strip()
        if not text:
            continue
        page_num = page.get("page_num", "")
        source = page.get("source", "Unknown source")
        title = page.get("title", "Uploaded PDF")
        
        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "title": title,
                    "heading": f"Page {page_num}",
                    "type": "pdf",
                    "page": page_num
                }
            )
            documents.append(doc)
            
    return documents
