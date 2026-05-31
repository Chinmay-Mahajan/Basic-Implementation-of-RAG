# Local RAG Chatbot with Query Rewriting

A fully local Retrieval-Augmented Generation (RAG) chatbot built with [LlamaIndex](https://github.com/run-llama/llama_index), [Ollama](https://ollama.com/), and HuggingFace embeddings. Ask questions over your own notes or documents — everything runs on your machine, no API keys required.

---

## Features

- **Local LLM** — Powered by Llama 3 (8B) via Ollama; no external API calls
- **Semantic Search** — Embeds document chunks using `sentence-transformers/all-MiniLM-L6-v2`
- **Query Rewriting** — A dedicated LLM agent rewrites and expands user queries before retrieval, improving search accuracy
- **Conversation Memory** — Maintains a rolling chat history buffer so follow-up questions resolve correctly
- **Configurable Chunking** — Tune `chunk_size` and `chunk_overlap` to fit your documents
- **Debug Mode** — Optionally display retrieved nodes and memory state after each turn

---

## Architecture

```
User Query
    │
    ▼
QueryRewriter (LLM)
    │  Strips noise, resolves pronouns using chat history
    │  Returns a list of refined sub-queries
    ▼
VectorStoreIndex Retriever
    │  Top-K nearest neighbour search over embedded chunks
    ▼
RAGpipeline.gen_response (LLM)
    │  Formats context + chat history into a prompt
    │  Generates a grounded answer
    ▼
ChatMemoryBuffer (updated)
```

---

## Requirements

- Python 3.9+
- [Ollama](https://ollama.com/) installed and running locally
- Llama 3 model pulled: `ollama pull llama3`

### Python Dependencies

```bash
pip install llama-index
pip install llama-index-llms-ollama
pip install llama-index-embeddings-huggingface
pip install sentence-transformers
```

Or install from a requirements file (see below).

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

### 5. Configure and run

In `main.py`, update the `dir_path` to point to your documents folder:

```python
ra = RAGpipeline(
    embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
    dir_path="./docs",        # <-- path to your notes/documents
    chunk_size=200,
    chunk_overlap=20,
    show_node=True,
    show_mem=True
)
```

Then run:

```bash
python main.py
```

---

## Usage

Once running, the chatbot enters an interactive loop:

```
ENTER THE PROMPT:__ What are the key ideas in my notes on transformers?
```

Type `quit` to exit.

The **QueryRewriter** automatically breaks your question into cleaner sub-queries before searching, which helps surface more relevant chunks — especially for vague or pronoun-heavy follow-ups like *"Can you expand on that?"*

---

##  Configuration

| Parameter | Default | Description |
|---|---|---|
| `embedding_model_name` | `all-MiniLM-L6-v2` | HuggingFace embedding model |
| `dir_path` | — | Path to your documents directory |
| `chunk_size` | `200` | Token size of each document chunk |
| `chunk_overlap` | `20` | Token overlap between adjacent chunks |
| `show_node` | `True` | Print retrieved chunks after each response |
| `show_mem` | `False` | Print full chat memory after each turn |

---

##  Project Structure

```
.
├── main.py          # Entry point — pipeline setup and chat loop
├── prompts.py       # Prompt templates (QA prompt, query rewriter prompt)
├── requirements.txt
└── README.md
```

---






