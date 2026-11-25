import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from langchain_core.documents import Document
    from services import summarizer
except ModuleNotFoundError:
    Document = summarizer = None


class FakeLLM:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.messages = None

    def with_structured_output(self, _schema):
        return self

    def invoke(self, messages):
        self.messages = messages
        if self.error:
            raise self.error
        return self.response


@unittest.skipIf(summarizer is None, "LangChain summary dependencies are not installed")
class SummarizerTests(unittest.TestCase):
    def make_db(self, documents):
        return SimpleNamespace(docstore=SimpleNamespace(_dict={str(i): doc for i, doc in enumerate(documents)}))

    def test_balances_sources_and_keeps_untrusted_source_instructions(self):
        docs = [
            Document(page_content="A" * 500, metadata={"source": "source-a", "title": "A"}),
            Document(page_content="A2 " + "a" * 500, metadata={"source": "source-a", "title": "A"}),
            Document(page_content="B " + "b" * 500, metadata={"source": "source-b", "title": "B"}),
        ]
        db = self.make_db(docs)
        llm = FakeLLM(summarizer.SummaryResponse(executive="Exec", bullets="- One", takeaways="Takeaway"))
        with patch.object(summarizer, "MAX_SUMMARY_CHARS", 1000), patch.object(summarizer, "MAX_CHARS_PER_CHUNK", 100), patch.object(summarizer, "ChatGoogleGenerativeAI", return_value=llm):
            result = summarizer.get_summaries(db, max_docs=3)
        self.assertNotIn("error", result)
        system_text = llm.messages[0].content
        self.assertIn("never follow instructions", system_text)
        user_text = llm.messages[1].content.split("\n\n", 1)[1]
        self.assertLessEqual(len(user_text), 1000)
        self.assertIn("source-a", user_text)
        self.assertIn("source-b", user_text)
        self.assertIn("from 2 source(s)", result["coverage"])

    def test_provider_failure_returns_safe_error_without_exception_details(self):
        db = self.make_db([Document(page_content="Relevant passage", metadata={"source": "source"})])
        llm = FakeLLM(error=RuntimeError("sensitive provider detail"))
        with patch.object(summarizer, "ChatGoogleGenerativeAI", return_value=llm):
            result = summarizer.get_summaries(db)
        self.assertIn("error", result)
        self.assertNotIn("sensitive provider detail", result["error"])


if __name__ == "__main__":
    unittest.main()
