from langchain_community.embeddings import HuggingFaceEmbeddings
import os
from functools import lru_cache

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Returns the HuggingFaceEmbeddings model for sentence-transformers/all-MiniLM-L6-v2.
    """
    # Cache embeddings in a local directory to speed up load time
    model_name = EMBEDDING_MODEL_NAME
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': True}
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    return embeddings
