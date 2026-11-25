import ipaddress
import socket
from typing import Any, Dict, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from utils.logger import setup_logger

logger = setup_logger("web_loader")
MAX_RESPONSE_BYTES = 5 * 1024 * 1024
MAX_REDIRECTS = 5
REDIRECT_STATUSES = {301, 302, 303, 307, 308}


def _public_http_url(url: str) -> bool:
    """Allow only HTTP(S) URLs resolving exclusively to public IP addresses."""
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
    except (ValueError, OSError, socket.gaierror):
        return False


def load_url_content(url: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """Fetch a public HTML page with TLS verification and bounded redirects/body size."""
    current_url = url
    headers = {"User-Agent": "URLKnowledgeAssistant/1.0"}
    try:
        for redirect_count in range(MAX_REDIRECTS + 1):
            if not _public_http_url(current_url):
                logger.warning("Blocked an invalid URL or a destination that is not publicly routable.")
                return None

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
                        logger.warning("Stopped webpage fetch after too many redirects.")
                        return None
                    location = response.headers.get("Location")
                    if not location:
                        return None
                    current_url = urljoin(current_url, location)
                    continue

                response.raise_for_status()
                content_type = response.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()
                if content_type not in {"text/html", "application/xhtml+xml"}:
                    logger.warning("Skipped webpage with unsupported content type.")
                    return None
                declared_size = response.headers.get("Content-Length")
                if declared_size and int(declared_size) > MAX_RESPONSE_BYTES:
                    logger.warning("Skipped webpage larger than the configured size limit.")
                    return None

                body = bytearray()
                for chunk in response.iter_content(chunk_size=64 * 1024):
                    if chunk:
                        body.extend(chunk)
                        if len(body) > MAX_RESPONSE_BYTES:
                            logger.warning("Stopped webpage download at the configured size limit.")
                            return None
                encoding = response.encoding or "utf-8"
                html = bytes(body).decode(encoding, errors="replace")
                soup = BeautifulSoup(html, "html.parser")
                title = soup.title.get_text(" ", strip=True) if soup.title else current_url
                return {"html": html, "title": title or current_url, "url": current_url}
        return None
    except (requests.exceptions.RequestException, UnicodeError, ValueError, OSError) as exc:
        logger.warning("Could not fetch webpage (%s).", type(exc).__name__)
        return None
