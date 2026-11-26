"""Runs the real Streamlit script headlessly. Skipped when Streamlit isn't installed."""
import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "streamlit not installed")
class AppSmokeTests(unittest.TestCase):
    def test_app_renders_without_errors(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=30).run()
        self.assertFalse(app.exception, app.exception)
        self.assertTrue(any("Ask your sources" in m.value for m in app.markdown))
        for view in ("Summary", "Sources", "Chat"):
            app.radio[0].set_value(view).run()
            self.assertFalse(app.exception, app.exception)


if __name__ == "__main__":
    unittest.main()
