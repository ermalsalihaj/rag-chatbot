import os
import sys
import json
from pathlib import Path
from glob import glob

# Add project root to Python path (Windows script execution)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

from rag.config import DOCS_DIR, INDEX_DIR, OPENAI_EMBEDDING_MODEL

CHUNKS_JSONL = os.path.join("data", "index", "chunks.jsonl")

def load_documents(docs_dir: str):
    docs = []

    pdfs = glob(os.path.join(docs_dir, "**/*.pdf"), recursive=True)
    txts = glob(os.path.join(docs_dir, "**/*.txt"), recursive=True)
    mds  = glob(os.path.join(docs_dir, "**/*.md"), recursive=True)

    for path in pdfs:
        docs.extend(PyPDFLoader(path).load())

    for path in txts:
        docs.extend(TextLoader(path, encoding="utf-8").load())

    for path in mds:
        docs.extend(TextLoader(path, encoding="utf-8").load())

    return docs

def save_chunks_jsonl(chunks):
    os.makedirs(os.path.dirname(CHUNKS_JSONL), exist_ok=True)
    with open(CHUNKS_JSONL, "w", encoding="utf-8") as f:
        for i, d in enumerate(chunks):
            rec = {
                "id": i,
                "text": d.page_content,
                "metadata": d.metadata or {},
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    docs = load_documents(DOCS_DIR)
    if not docs:
        raise SystemExit(f"No documents found in {DOCS_DIR}. Put PDFs/TXT/MD there first.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=OPENAI_EMBEDDING_MODEL)

    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(INDEX_DIR)

    save_chunks_jsonl(chunks)

    print(f"Loaded documents: {len(docs)}")
    print(f"Created chunks:   {len(chunks)}")
    print(f"Saved FAISS to:   {INDEX_DIR}")
    print(f"Saved chunks to:  {CHUNKS_JSONL}")

if __name__ == "__main__":
    main()
