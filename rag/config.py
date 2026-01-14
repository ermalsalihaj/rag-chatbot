from dotenv import load_dotenv
import os

load_dotenv()

DOCS_DIR = "data/raw/pubmed"
INDEX_DIR = "data/index/faiss"

TOP_K = 4

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
