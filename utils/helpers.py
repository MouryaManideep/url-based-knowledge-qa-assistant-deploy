import hashlib
import re
from urllib.parse import quote, urlparse
from typing import List, Dict

# File: utils/helpers.py
# Purpose: Shared utility functions for URL validation, text extraction, and chat export.

def is_valid_url(url: str) -> bool:
    """Validate HTTP(S) URL syntax; the loader separately checks network safety."""
    try:
        result = urlparse(url.strip())
        return (
            result.scheme.lower() in {"http", "https"}
            and bool(result.hostname)
            and not re.search(r"\s", result.hostname)
            and not result.username
            and not result.password
            and (result.port is None or 1 <= result.port <= 65535)
        )
    except (AttributeError, ValueError):
        return False


def content_fingerprint(data: bytes) -> str:
    """Return a stable SHA-256 fingerprint for binary document contents."""
    return hashlib.sha256(data).hexdigest()

def extract_urls(text: str) -> List[str]:
    """Extract all URLs from a block of text."""
    url_pattern = re.compile(
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return url_pattern.findall(text)

def export_chat_to_markdown(history: List[Dict[str, str]]) -> str:
    """Format the conversation history into a markdown string."""
    def escape_markdown(value: object) -> str:
        text = str(value).replace("\r", " ").replace("\n", " ")
        return re.sub(r"([\\`*_{}\[\]()#+.!|>])", r"\\\1", text)

    md_content = "# Chat History Export\n\n"
    for message in history:
        role = message.get("role", "Unknown").capitalize()
        content = message.get("content", "")
        md_content += f"### {role}\n{content}\n\n"
        
        # If citations are present, add them
        if "citations" in message and message["citations"]:
            md_content += "**Citations:**\n"
            for cite in message["citations"]:
                source = cite.get("source", "N/A")
                title = escape_markdown(cite.get("title", "N/A"))
                heading = escape_markdown(cite.get("heading", "N/A"))
                page = cite.get("page", "")
                page_str = f", Page {page}" if page else ""
                if is_valid_url(str(source)):
                    safe_source = quote(str(source), safe=":/?#[]@!$&'*,;=%")
                    md_content += f"- *{title}* (Section: {heading}{page_str}) - [Link]({safe_source})\n"
                else:
                    md_content += f"- *{title}* (Section: {heading}{page_str}) - Source: {escape_markdown(source)}\n"
            md_content += "\n"
            
    return md_content

def extract_text_content(content: any) -> str:
    """Extract clean string content from LangChain/Gemini response content."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, dict):
                if "text" in part:
                    text_parts.append(part["text"])
            elif isinstance(part, str):
                text_parts.append(part)
        return "".join(text_parts)
    return str(content)
