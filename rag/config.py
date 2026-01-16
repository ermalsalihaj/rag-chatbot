from dotenv import load_dotenv
import os

load_dotenv()

DOCS_DIR = "data/raw/pubmed"
INDEX_DIR = "data/index/faiss"

TOP_K = 4

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

# Hybrid search + reranking
HYBRID_ENABLED_DEFAULT = True
BM25_FETCH_K = 12
VECTOR_FETCH_K = 12

RERANK_ENABLED_DEFAULT = True
RERANK_TOP_N = 12 
