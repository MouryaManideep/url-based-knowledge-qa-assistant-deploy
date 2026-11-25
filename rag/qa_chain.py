import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from rag.prompt import SYSTEM_PROMPT, HYBRID_SYSTEM_PROMPT, QA_TEMPLATE
from rag.retrieval import semantic_search
from utils.logger import setup_logger
from utils.helpers import extract_text_content

# File: rag/qa_chain.py
# Purpose: Retrieve relevant chunks from the vector store and ask Gemini for an answer.

load_dotenv()
logger = setup_logger("qa_chain")

# Ensure API Key is configured before creating the LLM client.
api_key = os.getenv("GEMINI_API_KEY")

def get_llm(temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    """Instantiate the Gemini LLM."""
    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY is not set in the environment variables.")
        
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash",
        temperature=temperature,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

def generate_answer(
    question: str, 
    db: Any, 
    chat_history: List[Dict[str, str]], 
    k: int = 5,
    strict: bool = True,
    temperature: float = 0.1,
    search_type: str = "semantic_similarity",
    min_similarity: float = 0.20,
) -> Tuple[str, List[Document]]:
    """
    Retrieves relevant chunks from FAISS, builds the conversation prompt including
    history, and calls the Gemini model. Returns the answer string and retrieved docs.
    """
    if not db:
        return "Knowledge base is empty. Please enter URLs or upload PDFs to construct the database.", []
        
    logger.info("Running %s semantic retrieval (top_k=%s, min_similarity=%s)", search_type, k, min_similarity)
    retrieved_docs = semantic_search(
        db=db,
        query=question,
        k=k,
        min_similarity=min_similarity,
        search_type=search_type,
    )
    if not retrieved_docs:
        return (
            "I couldn't find a sufficiently similar passage in the loaded sources. "
            "Try rephrasing your question, lowering the minimum semantic similarity, or adding more relevant sources.",
            [],
        )
    
    # Format retrieved contexts
    context_str = ""
    for idx, doc in enumerate(retrieved_docs):
        src = doc.metadata.get("source", "N/A")
        title = doc.metadata.get("title", "N/A")
        heading = doc.metadata.get("heading", "N/A")
        page = doc.metadata.get("page", "")
        page_info = f", Page {page}" if page else ""
        context_str += f"[{idx+1}] Source: {src} | Title: {title} | Section: {heading}{page_info}\nContent: {doc.page_content}\n\n"
        
    # Build messages list
    sys_prompt = SYSTEM_PROMPT if strict else HYBRID_SYSTEM_PROMPT
    messages = [SystemMessage(content=sys_prompt)]
    
    # Keep only recent, bounded turns to limit prompt growth and stale instructions.
    for msg in chat_history[-12:]:
        role = msg.get("role")
        content = str(msg.get("content") or "")[:3000]
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
            
    # Add current question with contexts
    prompt_content = QA_TEMPLATE.format(context=context_str, question=question)
    messages.append(HumanMessage(content=prompt_content))
    
    try:
        llm = get_llm(temperature=temperature)
        response = llm.invoke(messages)
        answer = extract_text_content(response.content)
        return str(answer), retrieved_docs
    except Exception as e:
        logger.error("Gemini answer generation failed (%s).", type(e).__name__)
        return "I couldn't generate an answer right now. Please check your Gemini configuration and try again.", retrieved_docs
