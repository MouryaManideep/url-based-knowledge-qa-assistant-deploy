import unittest

from preprocessing.cleaner import clean_html
from utils.helpers import export_chat_to_markdown

try:
    from preprocessing.chunker import chunk_web_content, chunk_pdf_content
except ModuleNotFoundError:
    chunk_web_content = chunk_pdf_content = None


class CleanerTests(unittest.TestCase):
    def test_nested_list_text_is_not_duplicated(self):
        sections = clean_html("<html><body><h1>List</h1><ul><li><p>First item</p></li><li>Second item</li></ul></body></html>")
        texts = [section["text"] for section in sections]
        self.assertEqual(texts.count("- First item"), 1)
        self.assertEqual(texts.count("- Second item"), 1)

    def test_content_with_navigation_and_footer_words_is_preserved(self):
        html = "<main><p class='navigation-history'>Navigation history details</p><p class='footer-notes'>Important footer notes</p><nav>Menu</nav></main>"
        text = " ".join(section["text"] for section in clean_html(html))
        self.assertIn("Navigation history details", text)
        self.assertIn("Important footer notes", text)
        self.assertNotIn("Menu", text)

    def test_empty_html_returns_no_empty_section(self):
        self.assertEqual(clean_html("<html><body>  </body></html>"), [])


@unittest.skipIf(chunk_web_content is None, "langchain-text-splitters is not installed")
class ChunkerTests(unittest.TestCase):
    def test_web_and_pdf_metadata_survive_chunking(self):
        web_docs = chunk_web_content([{"text": "Useful web passage", "heading": "Overview"}], "https://example.org", "Example", chunk_size=12, chunk_overlap=2)
        pdf_docs = chunk_pdf_content([{"text": "Useful PDF passage", "page_num": 4, "source": "report.pdf", "title": "Report"}], chunk_size=12, chunk_overlap=2)
        self.assertTrue(web_docs)
        self.assertTrue(all(doc.metadata["source"] == "https://example.org" and doc.metadata["heading"] == "Overview" for doc in web_docs))
        self.assertTrue(pdf_docs)
        self.assertTrue(all(doc.metadata["page"] == 4 and doc.metadata["source"] == "report.pdf" for doc in pdf_docs))

    def test_invalid_overlap_and_empty_records(self):
        with self.assertRaises(ValueError):
            chunk_web_content([], "https://example.org", "Example", chunk_size=10, chunk_overlap=10)
        self.assertEqual(chunk_web_content([{"text": "  "}], "https://example.org", "Example"), [])
        self.assertEqual(chunk_pdf_content([{"text": ""}]), [])


class ExportSafetyTests(unittest.TestCase):
    def test_unsafe_citation_is_not_emitted_as_a_link(self):
        markdown = export_chat_to_markdown([{"role": "assistant", "content": "Answer", "citations": [{"title": "Bad *title*", "heading": "Section", "source": "javascript:alert(1)"}]}])
        self.assertNotIn("[Link](javascript:", markdown)
        self.assertIn(r"Source: javascript:alert\(1\)", markdown)
        self.assertIn(r"Bad \*title\*", markdown)

    def test_parentheses_in_safe_urls_are_encoded(self):
        markdown = export_chat_to_markdown([{"role": "assistant", "content": "Answer", "citations": [{"title": "Doc", "heading": "Heading", "source": "https://example.org/page_(one)"}]}])
        self.assertIn("https://example.org/page_%28one%29", markdown)


if __name__ == "__main__":
    unittest.main()
