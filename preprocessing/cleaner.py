from bs4 import BeautifulSoup
import re
from typing import Dict, Any, List

# File: preprocessing/cleaner.py
# Purpose: Remove noisy HTML parts like menus, footers, scripts, and ads before chunking.

def clean_html(html_content: str) -> List[Dict[str, Any]]:
    """
    Cleans raw HTML by stripping ads, scripts, navbars, sidebars, styles.
    Returns list of sections/paragraphs with their associated nearest headings.
    """
    soup = BeautifulSoup(html_content, "html.parser")
    
    # Remove HTML sections that usually do not contain the main document content.
    unwanted_tags = [
        "script", "style", "nav", "footer", "header", "aside", 
        "noscript", "iframe", "form", "button", "svg"
    ]
    for tag in soup(unwanted_tags):
        tag.decompose()
        
    # Match known boilerplate tokens, not broad substrings such as "nav" or "footer".
    noise_token = re.compile(r"(^|[-_])(ad|ads|advertisement|advertising|sidebar|navbar|nav-bar|menu|cookie|consent)([-_]|$)", re.I)
    for tag in list(soup.find_all(True)):
        tokens = list(tag.get("class", [])) + [tag.get("id", "")]
        if any(noise_token.search(str(token)) for token in tokens):
            tag.decompose()

    sections = []
    current_heading = "Introduction"
    
    # Traverse the page body and keep heading-based structure for easier retrieval.
    body = soup.body if soup.body else soup
    
    for element in body.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table"]):
        name = element.name
        # Parent list/table elements already include their descendants' text.
        if element.find_parent(["li", "table"]):
            continue
        text = element.get_text(" ", strip=True)
        if not text:
            continue
            
        if name.startswith("h"):
            current_heading = text
        elif name == "table":
            # Format tables as pipe-separated content for better LLM retrieval reading
            rows = []
            for row in element.find_all("tr"):
                cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
                if any(cells):
                    rows.append(" | ".join(cells))
            if rows:
                table_text = "\n".join(rows)
                sections.append({
                    "text": table_text,
                    "heading": current_heading
                })
        elif name == "li":
            sections.append({
                "text": f"- {text}",
                "heading": current_heading
            })
        else:
            sections.append({
                "text": text,
                "heading": current_heading
            })
            
    # If sections is empty but there is text, fallback to getting all paragraph texts
    if not sections:
        text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
        if text:
            sections.append({"text": text, "heading": "General Content"})
        
    return sections
