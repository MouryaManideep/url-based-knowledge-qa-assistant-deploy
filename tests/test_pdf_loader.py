import io
import unittest
from unittest.mock import patch

try:
    import fitz
except ModuleNotFoundError:
    fitz = None

if fitz is not None:
    from loaders import pdf_loader
else:
    pdf_loader = None


@unittest.skipIf(fitz is None, "PyMuPDF is not installed")
class PdfLoaderTests(unittest.TestCase):
    @staticmethod
    def make_pdf(page_texts):
        document = fitz.open()
        for text in page_texts:
            page = document.new_page()
            page.insert_text((72, 72), text)
        payload = document.tobytes()
        document.close()
        return payload

    def test_extracts_page_metadata(self):
        pages = pdf_loader.load_pdf_content(self.make_pdf(["First page", "Second page"]), "sample.pdf")
        self.assertEqual([page["page_num"] for page in pages], [1, 2])
        self.assertIn("First page", pages[0]["text"])
        self.assertEqual(pages[0]["source"], "sample.pdf")

    def test_rejects_oversized_input(self):
        with patch.object(pdf_loader, "MAX_PDF_BYTES", 4):
            self.assertEqual(pdf_loader.load_pdf_content(b"12345", "large.pdf"), [])

    def test_page_and_extracted_text_limits_return_no_partial_content(self):
        payload = self.make_pdf(["Page one text", "Page two text"])
        with patch.object(pdf_loader, "MAX_PDF_PAGES", 1):
            self.assertEqual(pdf_loader.load_pdf_content(payload, "many-pages.pdf"), [])
        with patch.object(pdf_loader, "MAX_EXTRACTED_CHARS", 1):
            self.assertEqual(pdf_loader.load_pdf_content(payload, "long-text.pdf"), [])

    def test_malformed_pdf_returns_no_content(self):
        self.assertEqual(pdf_loader.load_pdf_content(io.BytesIO(b"not a pdf"), "broken.pdf"), [])


if __name__ == "__main__":
    unittest.main()
