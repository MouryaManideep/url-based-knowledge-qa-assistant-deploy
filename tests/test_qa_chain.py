import unittest
from types import SimpleNamespace
from unittest.mock import patch

try:
    from langchain_core.documents import Document
    from rag import qa_chain
except ModuleNotFoundError:
    Document = qa_chain = None


class FakeLLM:
    def __init__(self, response="Grounded answer"):
        self.response = response
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return SimpleNamespace(content=self.response)


@unittest.skipIf(qa_chain is None, "LangChain/Gemini QA dependencies are not installed")
class QaChainTests(unittest.TestCase):
    def setUp(self):
        self.doc = Document(page_content="Relevant evidence", metadata={"source": "https://example.org", "title": "Example", "heading": "Facts"})

    def test_history_is_capped_and_context_is_added(self):
        history = [{"role": "user" if i % 2 == 0 else "assistant", "content": f"turn-{i}"} for i in range(20)]
        llm = FakeLLM()
        with patch.object(qa_chain, "semantic_search", return_value=[self.doc]), patch.object(qa_chain, "get_llm", return_value=llm):
            answer, docs = qa_chain.generate_answer("current question", object(), history)
        self.assertEqual(answer, "Grounded answer")
        self.assertEqual(docs, [self.doc])
        self.assertEqual(len(llm.messages), 14)  # system + 12 history turns + current prompt
        self.assertEqual(llm.messages[1].content, "turn-8")
        self.assertIn("Relevant evidence", llm.messages[-1].content)
        self.assertIn("current question", llm.messages[-1].content)

    def test_provider_errors_do_not_expose_exception_details(self):
        with patch.object(qa_chain, "semantic_search", return_value=[self.doc]), patch.object(qa_chain, "get_llm", side_effect=RuntimeError("secret provider response")):
            answer, docs = qa_chain.generate_answer("question", object(), [])
        self.assertNotIn("secret provider response", answer)
        self.assertEqual(docs, [self.doc])


if __name__ == "__main__":
    unittest.main()
