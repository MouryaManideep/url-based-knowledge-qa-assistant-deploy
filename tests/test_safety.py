import unittest
from unittest.mock import MagicMock, patch

from loaders import web_loader
from utils.helpers import content_fingerprint, is_valid_url


class URLValidationTests(unittest.TestCase):
    def test_accepts_http_and_https(self):
        self.assertTrue(is_valid_url("https://example.org/path"))
        self.assertTrue(is_valid_url("http://example.org"))

    def test_rejects_unsupported_or_malformed_urls(self):
        for value in ("", "ftp://example.org/file", "javascript:alert(1)", "https:///missing-host", "https://user:pass@example.org", "https://example.org:bad"):
            with self.subTest(value=value):
                self.assertFalse(is_valid_url(value))

    def test_blocks_loopback_before_request(self):
        with patch.object(web_loader.requests, "get") as get:
            self.assertIsNone(web_loader.load_url_content("http://127.0.0.1/admin"))
            get.assert_not_called()

    def test_blocks_private_dns_result_before_request(self):
        private_answer = [(None, None, None, None, ("10.0.0.8", 443))]
        with patch.object(web_loader.socket, "getaddrinfo", return_value=private_answer), patch.object(web_loader.requests, "get") as get:
            self.assertIsNone(web_loader.load_url_content("https://internal.example"))
            get.assert_not_called()

    def test_fetch_verifies_tls_and_returns_html_title(self):
        response = MagicMock()
        response.status_code = 200
        response.headers = {"Content-Type": "text/html; charset=utf-8", "Content-Length": "35"}
        response.encoding = "utf-8"
        response.iter_content.return_value = [b"<html><title>Example</title></html>"]
        response.__enter__.return_value = response
        with patch.object(web_loader.socket, "getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 443))]), patch.object(web_loader.requests, "get", return_value=response) as get:
            result = web_loader.load_url_content("https://example.org")
        self.assertEqual(result["title"], "Example")
        self.assertTrue(get.call_args.kwargs["verify"])
        self.assertFalse(get.call_args.kwargs["allow_redirects"])

    def test_redirect_to_loopback_is_blocked(self):
        response = MagicMock()
        response.status_code = 302
        response.headers = {"Location": "http://127.0.0.1/private"}
        response.__enter__.return_value = response
        with patch.object(web_loader.socket, "getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 443))]), patch.object(web_loader.requests, "get", return_value=response) as get:
            self.assertIsNone(web_loader.load_url_content("https://example.org/redirect"))
        self.assertEqual(get.call_count, 1)

    def test_rejects_oversized_response(self):
        response = MagicMock()
        response.status_code = 200
        response.headers = {"Content-Type": "text/html", "Content-Length": str(web_loader.MAX_RESPONSE_BYTES + 1)}
        response.__enter__.return_value = response
        with patch.object(web_loader.socket, "getaddrinfo", return_value=[(None, None, None, None, ("93.184.216.34", 443))]), patch.object(web_loader.requests, "get", return_value=response):
            self.assertIsNone(web_loader.load_url_content("https://example.org"))


class FingerprintTests(unittest.TestCase):
    def test_content_fingerprint_is_stable_and_content_based(self):
        self.assertEqual(content_fingerprint(b"same bytes"), content_fingerprint(b"same bytes"))
        self.assertNotEqual(content_fingerprint(b"first PDF"), content_fingerprint(b"different PDF"))


if __name__ == "__main__":
    unittest.main()
