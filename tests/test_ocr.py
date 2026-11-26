import unittest
from unittest.mock import patch

from rag.llm import OCR_TIMEOUT_S, reset_cooldowns
from rag.types import SourceError
from services import ingest
from services.ingest import ingest_file, split_ocr_pages
from tests.test_llm import client_with
from tests.test_loaders import make_pdf


class FakeOCR:
    def __init__(self, text):
        self.text, self.calls = text, []

    def __call__(self, data, mime_type):
        self.calls.append((len(data), mime_type))
        return self.text


class OCRIngestTests(unittest.TestCase):
    def test_split_ocr_pages(self):
        text = "=== Page 1 ===\nHostel fee refund\n\n=== Page 2 ===\n\n=== Page 3 ===\nSigned, Warden"
        self.assertEqual(split_ocr_pages(text), [(1, "Hostel fee refund"), (3, "Signed, Warden")])
        self.assertEqual(split_ocr_pages("No markers here"), [(1, "No markers here")])
        self.assertEqual(split_ocr_pages("   "), [])

    def test_scanned_pdf_without_key_explains_what_to_do(self):
        with self.assertRaises(SourceError) as ctx:
            ingest_file("cancel of hostel.pdf", make_pdf(["", ""]), 500, 50)
        self.assertIn("API key", str(ctx.exception))

    def test_scanned_pdf_is_read_with_ocr_and_keeps_page_numbers(self):
        ocr = FakeOCR("=== Page 1 ===\nRequest to cancel hostel room 214.\n=== Page 2 ===\nRefund of the deposit: Rs 5,000.")
        source, chunks = ingest_file("cancel of hostel.pdf", make_pdf(["", ""]), 500, 50, ocr=ocr)
        self.assertEqual(ocr.calls[0][1], "application/pdf")
        self.assertEqual(source.kind, "pdf")
        self.assertEqual([c.location for c in chunks], ["Page 1", "Page 2"])
        self.assertIn("Rs 5,000", chunks[1].text)

    def test_text_pdf_never_calls_ocr(self):
        ocr = FakeOCR("unused")
        ingest_file("notes.pdf", make_pdf(["Real text layer"]), 500, 50, ocr=ocr)
        self.assertEqual(ocr.calls, [])

    def test_photo_upload_uses_ocr(self):
        ocr = FakeOCR("Receipt\nTotal: 1,250")
        source, chunks = ingest_file("receipt.JPG", b"\xff\xd8 fake jpeg", 500, 50, ocr=ocr)
        self.assertEqual(ocr.calls[0][1], "image/jpeg")
        self.assertEqual(source.kind, "image")
        self.assertIn("1,250", chunks[0].text)
        with self.assertRaises(SourceError):
            ingest_file("receipt.png", b"png", 500, 50)  # No key: explained, not read.

    def test_long_scans_are_refused_before_calling_gemini(self):
        ocr = FakeOCR("unused")
        with patch.object(ingest, "MAX_OCR_PAGES", 1), self.assertRaises(SourceError) as ctx:
            ingest_file("long scan.pdf", make_pdf(["", ""]), 500, 50, ocr=ocr)
        self.assertIn("2 pages", str(ctx.exception))
        self.assertEqual(ocr.calls, [])

    def test_empty_ocr_result_is_an_error(self):
        with self.assertRaises(SourceError):
            ingest_file("blank.png", b"png", 500, 50, ocr=FakeOCR("=== Page 1 ===\n"))


class TranscribeTests(unittest.TestCase):
    def setUp(self):
        reset_cooldowns()

    def test_transcribe_sends_file_inline_with_longer_timeout(self):
        llm, models, _ = client_with(["=== Page 1 ===\nhello"])
        captured = {}
        original = models.generate_content

        def spy(model, contents, config):
            captured.update(contents=contents, config=config)
            return original(model, contents, config)

        models.generate_content = spy
        self.assertIn("hello", llm.transcribe(b"%PDF-bytes", "application/pdf"))
        parts = captured["contents"][0]["parts"]
        self.assertEqual(parts[0]["inline_data"], {"mime_type": "application/pdf", "data": b"%PDF-bytes"})
        self.assertIn("=== Page N ===", parts[1]["text"])
        self.assertEqual(captured["config"]["http_options"]["timeout"], OCR_TIMEOUT_S * 1000)


if __name__ == "__main__":
    unittest.main()
