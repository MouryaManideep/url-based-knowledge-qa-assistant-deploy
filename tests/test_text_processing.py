import unittest

from preprocessing.chunker import split_text
from preprocessing.cleaner import parse_html


class ChunkerTests(unittest.TestCase):
    def test_respects_size_and_overlap(self):
        text = " ".join(f"Sentence number {i} talks about topic {i % 7}." for i in range(300))
        chunks = split_text(text, 300, 60)
        self.assertGreater(len(chunks), 5)
        self.assertTrue(all(len(c) <= 300 for c in chunks))
        # Adjacent chunks share text when overlap is enabled.
        self.assertTrue(any(chunks[i][-20:].split()[-1] in chunks[i + 1] for i in range(len(chunks) - 1)))

    def test_short_and_empty_text(self):
        self.assertEqual(split_text("  hello   world  ", 100, 10), ["hello world"])
        self.assertEqual(split_text("   \n  ", 100, 10), [])

    def test_unbroken_text_is_hard_split(self):
        chunks = split_text("x" * 1000, 100, 0)
        self.assertEqual(len(chunks), 10)

    def test_rejects_bad_parameters(self):
        with self.assertRaises(ValueError):
            split_text("text", 100, 100)
        with self.assertRaises(ValueError):
            split_text("text", 10, 0)


class CleanerTests(unittest.TestCase):
    HTML = """<html><head><title>Guide</title><script>evil()</script></head><body>
    <nav>Home | About</nav><div class="cookie-banner">Accept cookies</div>
    <main><h1>Guide</h1><p>First paragraph about llamas and their wool, which is long enough to count.</p>
    <p>Second paragraph about alpacas grazing in the high Andes mountains every single day.</p>
    <h2>Prices</h2><table><tr><th>Item</th><th>Cost</th></tr><tr><td>Wool</td><td>$5</td></tr></table>
    <ul><li>Point one</li><li>Point two</li></ul></main>
    <footer>Copyright</footer></body></html>"""

    def test_extracts_title_sections_and_drops_boilerplate(self):
        title, sections = parse_html(self.HTML)
        self.assertEqual(title, "Guide")
        text = "\n".join(t for _, t in sections)
        for noise in ("evil()", "Home | About", "Accept cookies", "Copyright"):
            self.assertNotIn(noise, text)
        headings = [h for h, _ in sections]
        self.assertEqual(headings, ["Guide", "Prices"])
        self.assertIn("llamas", sections[0][1])
        self.assertIn("alpacas", sections[0][1])  # Paragraphs under one heading are grouped.
        self.assertIn("Wool | $5", sections[1][1])
        self.assertIn("- Point two", sections[1][1])

    def test_bytes_input_detects_meta_charset(self):
        html = '<html><head><meta charset="utf-8"><title>Café</title></head><body><p>Crème brûlée</p></body></html>'
        title, sections = parse_html(html.encode("utf-8"))
        self.assertEqual(title, "Café")
        self.assertIn("Crème brûlée", sections[0][1])

    def test_compound_state_classes_do_not_remove_the_page(self):
        # Wikipedia puts "...-main-menu-pinned-disabled" on <html>; that must not delete the article.
        html = ('<html class="client-js vector-feature-main-menu-pinned-disabled"><head><title>RAG</title></head>'
                '<body><div class="vector-main-menu-container">Menu</div><main class="mw-body"><h1>RAG</h1>'
                '<div class="mw-parser-output"><p>Retrieval-augmented generation lets models use external data.</p></div>'
                '</main></body></html>')
        title, sections = parse_html(html)
        self.assertEqual(title, "RAG")
        self.assertIn("external data", sections[0][1])
        self.assertNotIn("Menu", "\n".join(t for _, t in sections))

    def test_falls_back_when_filter_removes_everything(self):
        html = '<html><body><div class="sidebar"><p>Only content lives in a sidebar-classed div.</p></div></body></html>'
        self.assertIn("Only content", parse_html(html)[1][0][1])


if __name__ == "__main__":
    unittest.main()
