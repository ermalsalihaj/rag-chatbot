from typing import Dict, Any, List, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rag.config import INDEX_DIR, TOP_K

def load_db():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)

def format_citations(docs) -> List[str]:
    cites = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        page = d.metadata.get("page", None)
        if page is None:
            cites.append(f"{src}")
        else:
            cites.append(f"{src} (page {page})")
    return cites

def answer_question(question: str, min_score: float = 0.35) -> Dict[str, Any]:
    """
    min_score: heuristic threshold. FAISS distance/score scales vary,
    so we use it mainly as a 'weak retrieval' guard.
    """
    db = load_db()

    # returns [(Document, score)] where score is similarity score (implementation-specific)
    results: List[Tuple] = db.similarity_search_with_score(question, k=TOP_K)
    docs = [d for (d, s) in results]
    scores = [float(s) for (d, s) in results]

    # If nothing retrieved, or retrieval seems weak => safe response
    if not docs:
        return {"answer": "I don’t know based on the provided documents.", "citations": []}

    # Heuristic: if all scores indicate weak match, refuse
    # NOTE: depending on FAISS wrapper, lower can be better; we keep it simple for now.
    # We'll tune this after you test with your PDFs.
    if all(s > min_score for s in scores):
        return {"answer": "I don’t know based on the provided documents.", "citations": format_citations(docs)}

    # Grounded “extractive-ish” answer for now (safe): show top excerpts
    context = "\n\n".join([f"EXCERPT {i+1}:\n{docs[i].page_content}" for i in range(len(docs))])

    answer = (
        "Here are the most relevant excerpts I found in the documents. "
        "Ask a more specific question if you want a tighter answer:\n\n"
        + context[:2500]
    )

    return {"answer": answer, "citations": format_citations(docs)}
