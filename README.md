# Local RAG Chatbot with Query Rewriting

A fully local Retrieval-Augmented Generation (RAG) system built with [LlamaIndex](https://github.com/run-llama/llama_index), [Ollama](https://ollama.com/), and HuggingFace embeddings. Ask questions over your own notes or documents — everything runs on your machine, no API keys required.

Available in two modes: an interactive CLI for local use and a FastAPI server for programmatic access.

---

## Features

- **Local LLM** — Powered by Llama 3 (8B) via Ollama; no external API calls
- **Semantic Search** — Embeds document chunks using `sentence-transformers/all-MiniLM-L6-v2`
- **Query Rewriting** — A dedicated LLM pass rewrites and expands user queries before retrieval, improving search accuracy and resolving pronouns using chat history
- **Manual Reranking** — Retrieved chunks are re-scored using cosine similarity against the rewritten query to surface the most relevant context
- **Conversation Memory** — Maintains a rolling chat history buffer so follow-up questions resolve correctly
- **Persistent Index** — Embeddings are computed once and saved to disk; subsequent runs skip re-indexing
- **Two Interfaces** — Run as an interactive CLI or expose a REST endpoint via FastAPI

---

## Architecture

```
User Query
    |
    v
QueryRewriter (LLM)
    |  Resolves pronouns using chat history
    |  Returns a list of refined sub-queries
    v
VectorStoreIndex Retriever
    |  Top-10 nearest neighbour search per sub-query
    v
Manual Reranker (Cosine Similarity)
    |  Re-scores candidates, keeps top-3 per sub-query
    v
RAGpipeline.gen_response (LLM)
    |  Formats context + chat history into a prompt
    |  Generates a grounded answer
    v
ChatMemoryBuffer (updated)
```

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally
- Llama 3 pulled: `ollama pull llama3`

### Python Dependencies

```bash
pip install llama-index
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface
pip install sentence-transformers
pip install fastapi uvicorn   # only needed for the API server
```

Or install everything at once:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```
.
├── main.py          # Core pipeline: RAGpipeline, QueryRewriter, run_pipeline
├── ingestion.py     # Document loading, chunking, embedding, and index persistence
├── prompts.py       # Prompt templates for QA and query rewriting
├── api.py           # FastAPI app exposing the /query endpoint
├── storage/         # Persisted vector index (auto-generated on first run)
├── requirements.txt
└── README.md
```

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Chinmay-Mahajan/Basic-Implementation-of-RAG
cd Basic-Implementation-of-RAG
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama and pull the model

```bash
ollama serve
ollama pull llama3
```

### 4. Add your documents

Place your `.txt`, `.pdf`, or other supported files into a directory (e.g., `./docs`).

---

## CLI Mode

Configure the pipeline in `main.py`:

```python
ra = RAGpipeline(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    dir_path="./docs",   # path to your documents
    chunk_size=200,
    chunk_overlap=20,
    show_node=True,
    show_mem=False
)
```

Then run:

```bash
python main.py
```

Once running, enter queries at the prompt:

```
ENTER PROMPT: What are the key ideas in my notes on transformers?
```

Type `quit` to exit.

---

## API Mode

Start the FastAPI server:

```bash
uvicorn api:app --reload
```

On first run, documents are chunked, embedded, and saved to `./storage/`. Subsequent runs load the index from disk.

Query the endpoint:

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What are the key ideas in my notes on transformers?"}'
```

Response:

```json
{
  "answer": "...",
  "rewritten_queries": ["...", "..."],
  "retrieved_chunks": ["...", "..."]
}
```

---

## Configuration

| Parameter | Default | Description |
|---|---|---|
| `embedding_model_name` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `dir_path` | — | Path to your documents directory |
| `chunk_size` | `200` | Token size of each document chunk |
| `chunk_overlap` | `20` | Token overlap between adjacent chunks |
| `show_node` | `True` | Print retrieved chunks after each response |
| `show_mem` | `False` | Print full chat memory after each turn |

---

## Notes

- The `QueryRewriter` includes a retry loop (up to 3 attempts) to handle cases where the LLM returns malformed JSON.
- LlamaIndex's built-in `as_chat_engine` is initialised but not used for generation. Inspection revealed it duplicates conversation entries in chat history, degrading response quality. Generation is handled manually via `gen_response` instead.
