import os
from glob import glob

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rag.config import DOCS_DIR, INDEX_DIR

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

def main():
    os.makedirs(INDEX_DIR, exist_ok=True)

    docs = load_documents(DOCS_DIR)
    if not docs:
        raise SystemExit(f"No documents found in {DOCS_DIR}. Put PDFs there first.")

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectordb = FAISS.from_documents(chunks, embeddings)
    vectordb.save_local(INDEX_DIR)

    print(f"Loaded documents: {len(docs)}")
    print(f"Created chunks:   {len(chunks)}")
    print(f"Saved FAISS to:   {INDEX_DIR}")

if __name__ == "__main__":
    main()
