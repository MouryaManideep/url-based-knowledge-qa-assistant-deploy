"""Fetch public web pages with SSRF protections, TLS verification and size limits."""
from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

import requests

from config import MAX_WEB_BYTES
from rag.types import SourceError
from utils.logger import setup_logger

logger = setup_logger("web_loader")
MAX_RESPONSE_BYTES = MAX_WEB_BYTES
MAX_REDIRECTS = 5
REDIRECT_STATUSES = {301, 302, 303, 307, 308}
USER_AGENT = "Mozilla/5.0 (compatible; KnowledgeQA/2.0; +https://github.com/MouryaManideep)"

CONTENT_KINDS = {
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/plain": "text",
    "text/markdown": "text",
    "application/pdf": "pdf",
}


@dataclass
class FetchedPage:
    url: str
    kind: str  # "html", "text" or "pdf"
    body: bytes
    encoding: str | None


def _public_http_url(url: str) -> bool:
    """Allow only HTTP(S) URLs whose host resolves exclusively to public IP addresses."""
    try:
        parsed = urlparse(url)
        scheme = parsed.scheme.lower()
        if scheme not in {"http", "https"} or not parsed.hostname:
            return False
        if parsed.username or parsed.password:
            return False

        host = parsed.hostname.rstrip(".")
        if host.lower() == "localhost" or host.lower().endswith(".localhost"):
            return False
        port = parsed.port or (443 if scheme == "https" else 80)

        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            answers = socket.getaddrinfo(host, port, type=socket.SOCK_STREAM)
            addresses = [ipaddress.ip_address(answer[4][0].split("%", 1)[0]) for answer in answers]
        return bool(addresses) and all(address.is_global for address in addresses)
    except (ValueError, OSError):
        return False


def fetch_url(url: str, timeout: int = 15) -> FetchedPage:
    """Download a public page. Raises SourceError with a user-facing reason on failure."""
    current_url = url
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,application/pdf,text/plain;q=0.9,*/*;q=0.5"}
    try:
        for redirect_count in range(MAX_REDIRECTS + 1):
            if not _public_http_url(current_url):
                raise SourceError("This address is not a public web page.")

            with requests.get(
                current_url,
                headers=headers,
                timeout=(5, timeout),
                verify=True,
                allow_redirects=False,
                stream=True,
            ) as response:
                if response.status_code in REDIRECT_STATUSES:
                    if redirect_count >= MAX_REDIRECTS:
                        raise SourceError("The page redirected too many times.")
                    location = response.headers.get("Location")
                    if not location:
                        raise SourceError("The page sent an invalid redirect.")
                    current_url = urljoin(current_url, location)
                    continue

                if response.status_code in (401, 403):
                    raise SourceError("The site refused access (it may require a login or block bots).")
                if response.status_code == 404:
                    raise SourceError("Page not found (404).")
                if response.status_code >= 400:
                    raise SourceError(f"The site returned an error ({response.status_code}).")

                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                kind = CONTENT_KINDS.get(content_type)
                if kind is None:
                    raise SourceError(f"Unsupported content type ({content_type or 'unknown'}).")
                declared_size = response.headers.get("Content-Length")
                if declared_size and declared_size.isdigit() and int(declared_size) > MAX_RESPONSE_BYTES:
                    raise SourceError("The page is larger than the 5 MB limit.")

                body = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            raise SourceError("The page is larger than the 5 MB limit.")
                # requests guesses ISO-8859-1 for text/* without a charset; only trust an explicit one.
                has_charset = "charset=" in response.headers.get("Content-Type", "").lower()
                encoding = response.encoding if has_charset else None
                return FetchedPage(url=current_url, kind=kind, body=bytes(body), encoding=encoding)
        raise SourceError("The page redirected too many times.")
    except requests.exceptions.SSLError as exc:
        raise SourceError("The site's TLS certificate could not be verified.") from exc
    except requests.exceptions.Timeout as exc:
        raise SourceError("The site took too long to respond.") from exc
    except requests.exceptions.RequestException as exc:
        logger.warning("Could not fetch webpage (%s).", type(exc).__name__)
        raise SourceError("The site could not be reached.") from exc
