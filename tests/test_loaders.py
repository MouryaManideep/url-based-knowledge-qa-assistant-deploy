import unittest
from unittest.mock import MagicMock, patch

from loaders import web_loader
from loaders.pdf_loader import load_pdf
from rag.types import SourceError
from services.ingest import ingest_file, ingest_url, normalize_url, url_source_id

PUBLIC = [(None, None, None, None, ("93.184.216.34", 443))]


def make_pdf(pages):
    """Build a tiny valid PDF with one text line per page."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", None, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"]
    kids = []
    for text in pages:
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects.append(f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream")
        content_id = len(objects)
        objects.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 3 0 R >> >> /Contents {content_id} 0 R >>")
        kids.append(f"{len(objects)} 0 R")
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(kids)} >>"
    out, offsets = "%PDF-1.4\n", []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out.encode()))
        out += f"{number} 0 obj\n{body}\nendobj\n"
    xref = len(out.encode())
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n" + "".join(f"{o:010d} 00000 n \n" for o in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n"
    return out.encode()


def response(status=200, headers=None, body=b""):
    r = MagicMock()
    r.status_code = status
    r.headers = headers or {}
    r.encoding = "utf-8"
    r.iter_content.return_value = [body]
    r.__enter__.return_value = r
    return r


class WebLoaderSafetyTests(unittest.TestCase):
    def test_blocks_loopback_before_request(self):
        with patch.object(web_loader.requests, "get") as get, self.assertRaises(SourceError):
            web_loader.fetch_url("http://127.0.0.1/admin")
        get.assert_not_called()

    def test_blocks_private_dns_result(self):
        private = [(None, None, None, None, ("10.0.0.8", 443))]
        with patch.object(web_loader.socket, "getaddrinfo", return_value=private), \
                patch.object(web_loader.requests, "get") as get, self.assertRaises(SourceError):
            web_loader.fetch_url("https://internal.example")
        get.assert_not_called()

    def test_redirect_to_loopback_is_blocked(self):
        redirect = response(302, {"Location": "http://127.0.0.1/private"})
        with patch.object(web_loader.socket, "getaddrinfo", return_value=PUBLIC), \
                patch.object(web_loader.requests, "get", return_value=redirect) as get, self.assertRaises(SourceError):
            web_loader.fetch_url("https://example.org/redirect")
        self.assertEqual(get.call_count, 1)

    def test_rejects_oversized_and_unsupported(self):
        big = response(200, {"Content-Type": "text/html", "Content-Length": str(web_loader.MAX_RESPONSE_BYTES + 1)})
        image = response(200, {"Content-Type": "image/png"})
        for resp in (big, image):
            with patch.object(web_loader.socket, "getaddrinfo", return_value=PUBLIC), \
                    patch.object(web_loader.requests, "get", return_value=resp), self.assertRaises(SourceError):
                web_loader.fetch_url("https://example.org")

    def test_fetch_verifies_tls_and_ignores_guessed_charset(self):
        page = response(200, {"Content-Type": "text/html"}, b"<html><title>Example</title></html>")
        with patch.object(web_loader.socket, "getaddrinfo", return_value=PUBLIC), \
                patch.object(web_loader.requests, "get", return_value=page) as get:
            result = web_loader.fetch_url("https://example.org")
        self.assertEqual(result.kind, "html")
        self.assertIsNone(result.encoding)
        self.assertTrue(get.call_args.kwargs["verify"])
        self.assertFalse(get.call_args.kwargs["allow_redirects"])


class IngestTests(unittest.TestCase):
    def test_ingest_html_url(self):
        html = b"<html><title>Llamas</title><body><h2>Diet</h2><p>Llamas eat grass and hay.</p></body></html>"
        page = response(200, {"Content-Type": "text/html; charset=utf-8"}, html)
        with patch.object(web_loader.socket, "getaddrinfo", return_value=PUBLIC), \
                patch.object(web_loader.requests, "get", return_value=page):
            source, chunks = ingest_url("https://example.org/llamas/", 500, 50)
        self.assertEqual(source.title, "Llamas")
        self.assertEqual(chunks[0].location, "Diet")
        self.assertEqual(chunks[0].url, "https://example.org/llamas/")
        self.assertEqual(source.id, url_source_id("https://EXAMPLE.org/llamas#top"))

    def test_ingest_pdf_url(self):
        page = response(200, {"Content-Type": "application/pdf"}, make_pdf(["Remote PDF text"]))
        with patch.object(web_loader.socket, "getaddrinfo", return_value=PUBLIC), \
                patch.object(web_loader.requests, "get", return_value=page):
            source, chunks = ingest_url("https://example.org/report.pdf", 500, 50)
        self.assertEqual(source.kind, "pdf")
        self.assertEqual(chunks[0].page, 1)

    def test_ingest_pdf_and_text_files(self):
        source, chunks = ingest_file("report.pdf", make_pdf(["Alpha page", "Beta page"]), 500, 50)
        self.assertEqual([c.location for c in chunks], ["Page 1", "Page 2"])
        self.assertIn("Beta", chunks[1].text)
        source, chunks = ingest_file("notes.md", "# Notes\nSome *notes* here.".encode(), 500, 50)
        self.assertEqual(source.kind, "text")
        with self.assertRaises(SourceError):
            ingest_file("image.png", b"\x89PNG", 500, 50)

    def test_invalid_pdf_is_a_friendly_error(self):
        with self.assertRaises(SourceError):
            load_pdf(b"not a pdf", "x.pdf")
        with self.assertRaises(SourceError):  # A PDF with no extractable text.
            load_pdf(make_pdf([""]), "blank.pdf")

    def test_normalize_url(self):
        self.assertEqual(normalize_url("HTTPS://Example.org/a/#x"), "https://example.org/a")


if __name__ == "__main__":
    unittest.main()
