import os
import tempfile
import unittest
from unittest.mock import patch

try:
    import faiss
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_community.vectorstores import FAISS
    from vectorstore import faiss_store
except ModuleNotFoundError:
    faiss = Document = Embeddings = FAISS = faiss_store = None


if Embeddings is not None:
    class TinyEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [[float(len(text)), float(sum(map(ord, text)) % 13)] for text in texts]

        def embed_query(self, text):
            return [float(len(text)), float(sum(map(ord, text)) % 13)]


@unittest.skipIf(faiss_store is None, "FAISS/LangChain dependencies are not installed")
class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.index_dir = os.path.join(self.temp.name, "faiss_index")
        self.paths = patch.multiple(
            faiss_store,
            INDEX_DIR=self.index_dir,
            STORE_PATH=os.path.join(self.index_dir, "store.json"),
            MANIFEST_PATH=os.path.join(self.index_dir, "manifest.json"),
            EMBEDDING_MODEL_NAME="test-embedding",
        )
        self.paths.start()
        self.addCleanup(self.paths.stop)
        self.addCleanup(self.temp.cleanup)
        self.embeddings = TinyEmbeddings()

    def test_json_metadata_round_trip(self):
        document = Document(page_content="A semantic passage", metadata={"source": "https://example.org", "page": 3})
        with patch.object(faiss_store, "get_embedding_model", return_value=self.embeddings):
            saved = faiss_store.get_faiss_index([document])
            loaded = faiss_store.get_faiss_index()
        self.assertEqual(loaded.index.ntotal, 1)
        restored = next(iter(loaded.docstore._dict.values()))
        self.assertEqual(restored.page_content, document.page_content)
        self.assertEqual(restored.metadata, document.metadata)
        self.assertTrue(os.path.exists(os.path.join(self.index_dir, "manifest.json")))
        self.assertFalse(os.path.exists(os.path.join(self.index_dir, "index.pkl")))

    def test_legacy_pickle_index_is_not_loaded(self):
        os.makedirs(self.index_dir)
        with open(os.path.join(self.index_dir, "index.pkl"), "wb") as stream:
            stream.write(b"not loaded")
        with self.assertRaisesRegex(RuntimeError, "legacy pickle"):
            faiss_store._load_safe(self.embeddings, "test-embedding")

    def test_failed_save_keeps_the_previous_index_intact(self):
        original = FAISS.from_documents(
            [Document(page_content="original", metadata={"source": "old"})], self.embeddings
        )
        faiss_store._save_local(original)
        replacement = FAISS.from_documents(
            [Document(page_content="replacement", metadata={"source": "new"})], self.embeddings
        )
        with patch.object(faiss_store.faiss, "write_index", side_effect=OSError("simulated disk failure")):
            with self.assertRaises(OSError):
                faiss_store._save_local(replacement)
        restored = faiss_store._load_safe(self.embeddings, "test-embedding")
        self.assertEqual(next(iter(restored.docstore._dict.values())).page_content, "original")


if __name__ == "__main__":
    unittest.main()
