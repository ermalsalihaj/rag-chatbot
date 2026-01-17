# RAG Chatbot - FAISS + LangChain + OpenAI

A production-ready Retrieval-Augmented Generation (RAG) chatbot that answers questions using a curated set of healthcare documents. The chatbot features hybrid search, reranking, conversation memory, metadata filters, and comprehensive observability.

## Features

### Must-Have ✅
- **Document Ingestion Pipeline**: Load, chunk, embed, and index PDF/TXT/MD documents
- **Vector Search Retriever**: FAISS-based semantic search with Top-K retrieval
- **Answer Generation with Citations**: OpenAI-powered responses with source citations
- **Simple UI**: Streamlit-based chat interface with observability dashboard
- **Safe "I don't know" Handling**: Prevents hallucinations when information is not available

### Nice-to-Have ✅ (All Implemented)
- **Hybrid Search**: Combines BM25 (keyword) and FAISS (semantic) search for better retrieval
- **Reranking**: LLM-based reranking to improve document relevance
- **Metadata Filters**: Filter by source file and document type (PDF/TXT/MD)
- **Conversation Memory**: Short-term conversation context for better follow-up questions
- **Guardrails**: Prompt injection detection + malicious document filtering
- **Observability Dashboard**: Real-time logging with latency metrics and query analytics

## Tech Stack

- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Keyword Search**: BM25 (via `rank_bm25`)
- **Framework**: LangChain
- **Embeddings**: OpenAI (text-embedding-3-small)
- **LLM**: OpenAI GPT-4o-mini (configurable)
- **UI**: Streamlit with tabs and sidebar controls
- **Language**: Python 3.11+

## Project Structure

```
rag-chatbot/
├── app/
│   └── streamlit_app.py      # Streamlit chat UI with observability tab
├── rag/
│   ├── __init__.py
│   ├── config.py             # Configuration settings
│   ├── ingest_faiss.py       # Document ingestion pipeline
│   ├── rag_faiss.py          # RAG retrieval and generation (hybrid + reranking)
│   ├── safety.py             # Prompt injection detection
│   └── observability.py      # Logging utilities
├── data/
│   ├── raw/
│   │   └── pubmed/           # Input PDF documents (35+ documents)
│   ├── index/
│   │   ├── faiss/            # FAISS vector index
│   │   └── chunks.jsonl      # BM25 index (JSONL format)
│   └── observability/
│       └── rag_logs.csv      # Query logs with metrics
├── requirements.txt
└── README.md
```

## Setup

### Prerequisites

- Python 3.11 or higher
- OpenAI API key

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/ermalsalihaj/rag-chatbot.git
   cd rag-chatbot
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   # Optional: customize models
   OPENAI_MODEL=gpt-4o-mini
   OPENAI_EMBEDDING_MODEL=text-embedding-3-small
   ```

5. **Add documents**
   
   Place your PDF, TXT, or MD documents in `data/raw/pubmed/` directory. The project currently includes 35+ healthcare research papers.

6. **Run document ingestion**
   ```bash
   python rag/ingest_faiss.py
   ```
   
   This will:
   - Load all PDFs, TXT, and MD files from `data/raw/pubmed/`
   - Split documents into chunks (900 chars, 150 overlap)
   - Generate embeddings using OpenAI
   - Create and save FAISS index to `data/index/faiss/`
   - Create BM25 index (chunks.jsonl) for hybrid search

## How to Run

### Start the Streamlit App

```bash
streamlit run app/streamlit_app.py
```

The app will open in your browser at `http://localhost:8501`

### Usage

#### Chat Tab
1. **Ask Questions**: Type your question in the chat input
2. **View Answers**: The chatbot provides concise answers grounded in the documents
3. **Check Citations**: Each answer includes citations showing the source document and page number
4. **Handle Unknowns**: If the answer isn't in the documents, the bot says "I don't know based on the provided documents." without showing citations

#### Sidebar Controls
- **Search Mode**: Choose between "Hybrid" (BM25 + Vector) or "Vector" only
- **Enable Reranking**: Toggle LLM-based reranking for better relevance
- **Doc Type Filter**: Filter by document type (All, PDF, TXT, MD)
- **Metadata Filter**: Filter by source filename (e.g., type "CFIR" to only search CFIR documents)
- **Conversation Memory Turns**: Set how many previous Q/A pairs to remember (0-3)

#### Observability Tab
- View recent queries with full metadata
- Monitor latency metrics (retrieval, reranking, generation, total)
- Analyze search mode usage
- Track token usage estimates

### Example Questions

- "What are the main challenges of AI in healthcare?"
- "How did healthcare supply chains adapt during COVID-19?"
- "What are the barriers to using clinical decision support systems?"
- Follow-up: "Can you tell me more about that?" (uses conversation context)

## Configuration

Edit `rag/config.py` to customize:

### Retrieval Settings
- `TOP_K`: Number of document chunks to return (default: 4)
- `BM25_FETCH_K`: Number of candidates from BM25 search (default: 12)
- `VECTOR_FETCH_K`: Number of candidates from vector search (default: 12)

### Reranking Settings
- `RERANK_ENABLED_DEFAULT`: Enable reranking by default (default: True)
- `RERANK_TOP_N`: Number of documents to rerank (default: 12)

### OpenAI Settings
- `OPENAI_MODEL`: Chat model (default: "gpt-4o-mini")
- `OPENAI_EMBEDDING_MODEL`: Embedding model (default: "text-embedding-3-small")

### Paths
- `DOCS_DIR`: Directory containing input documents (default: "data/raw/pubmed")
- `INDEX_DIR`: Directory for FAISS index (default: "data/index/faiss")

## Features in Detail

### Document Ingestion

The ingestion pipeline (`rag/ingest_faiss.py`):
- Supports PDF, TXT, and MD files
- Uses RecursiveCharacterTextSplitter (chunk_size=900, chunk_overlap=150)
- Generates embeddings using OpenAI
- Creates FAISS vector index for semantic search
- Creates BM25 index (JSONL) for keyword search
- Stores chunk metadata (source, page number)

### Hybrid Search

The hybrid search system combines:
- **Vector Search (FAISS)**: Semantic similarity using embeddings
- **BM25 Search**: Keyword-based relevance using TF-IDF

Results are merged, deduplicated, and optionally reranked for optimal relevance.

### Reranking

LLM-based reranking:
- Takes top N candidates from hybrid search
- Uses OpenAI to rank by relevance to the question
- Returns best documents first
- Improves answer quality significantly

### Conversation Memory

Short-term conversation context:
- Maintains last N question-answer exchanges (configurable, 0-3)
- Includes previous context in prompts for better follow-up answers
- Keeps retrieval grounded (always retrieves from documents, never uses previous answers as facts)
- Helps with follow-up questions like "Can you tell me more?"

### Metadata Filters

Filter documents by:
- **Source**: Filter by filename (e.g., "CFIR" matches files containing "CFIR")
- **File Type**: Filter by document type (PDF, TXT, MD, or All)

Useful for focusing searches on specific documents or document types.

### Safety Features

**Prompt Injection Detection**:
- Detects common injection patterns in user questions
- Blocks attempts to override system instructions
- Returns safe error message

**Malicious Document Filtering**:
- Scans retrieved documents for instruction-like content
- Filters out suspicious chunks that might contain prompt injections
- Prevents documents from hijacking the system prompt

**Instruction Defense**:
- System prompt explicitly tells the model to ignore instructions in documents
- Safe "I don't know" handling with no citations

### Observability

Comprehensive logging system:
- **Query Logging**: Every question is logged with full metadata
- **Performance Metrics**: Tracks retrieval, reranking, and generation latency
- **Token Estimation**: Estimates prompt and answer token usage
- **Mode Tracking**: Records which search mode was used
- **Filter Tracking**: Records applied filters
- **CSV Export**: All logs saved to `data/observability/rag_logs.csv`

Dashboard features:
- Recent queries table
- Latency charts (last 100 queries)
- Mode usage statistics
- Real-time updates as you use the chatbot

## Dataset

The project uses 35+ healthcare research papers covering topics such as:
- AI applications in healthcare
- Healthcare supply chains and COVID-19
- Clinical informatics and EHR
- Healthcare sustainability
- Quality improvement and patient care
- Telemedicine and digital health
- Healthcare leadership and management

## Development

### Project Structure

- `/app`: Streamlit chat interface with observability dashboard
- `/rag`: Core RAG functionality
  - `ingest_faiss.py`: Document ingestion
  - `rag_faiss.py`: Hybrid search, reranking, and generation
  - `safety.py`: Security guardrails
  - `observability.py`: Logging utilities
  - `config.py`: Configuration
- `/data`: Documents, indexes, and logs

### Adding New Documents

1. Place PDF, TXT, or MD files in `data/raw/pubmed/`
2. Run `python rag/ingest_faiss.py` to update both indexes
3. Restart the Streamlit app

### Code Architecture

- **Lazy Loading**: FAISS and BM25 indexes are loaded once and cached
- **Deduplication**: Removes duplicate chunks from hybrid search results
- **Error Handling**: Graceful fallbacks if BM25 or reranking fails
- **Performance**: Optimized for low latency with caching

## Troubleshooting

### "No module named 'rag'"
- Make sure you're running commands from the project root directory
- The ingestion script includes path setup for proper imports

### "OPENAI_API_KEY is missing"
- Create a `.env` file with your OpenAI API key
- Ensure `python-dotenv` is installed

### "Missing chunks.jsonl"
- Run `python rag/ingest_faiss.py` to create the BM25 index
- Required for hybrid search mode

### Index not found
- Run `python rag/ingest_faiss.py` to create the indexes first
- Ensure `data/index/faiss/` directory exists

### BM25 search fails in hybrid mode
- The system automatically falls back to vector-only search
- Check that `chunks.jsonl` exists and is valid

### Reranking is slow
- Reranking uses LLM calls, which adds latency
- You can disable it in the sidebar if speed is more important than quality

## Performance Tips

1. **Use Vector-only mode** for faster queries (no BM25 overhead)
2. **Disable reranking** if latency is critical
3. **Reduce TOP_K** in config for faster retrieval
4. **Use metadata filters** to narrow search space
5. **Adjust memory turns** to balance context vs. prompt size

## License

This project is part of Giga Academy Cohort IV.

## Contact

For questions or issues, please contact the project maintainer or open an issue on GitHub.

## Acknowledgments

- Built with LangChain, FAISS, and OpenAI
- Uses rank-bm25 for keyword search
- Streamlit for the user interface
