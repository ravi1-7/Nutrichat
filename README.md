# NutriChat

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about nutrition using a textbook as its knowledge base. Ask a question, and the app retrieves relevant passages from the book and generates a grounded answer with source citations.

## How It Works

1. **Ingest** — Extract text from a nutrition PDF, chunk it, generate embeddings, and store vectors in Supabase (pgvector)
2. **Retrieve** — Embed the user's query, perform semantic search against stored chunks
3. **Generate** — Feed retrieved context to an LLM to produce an answer with page references

## Tech Stack

| Layer | Technology |
|-------|-----------|
| PDF Extraction | PyMuPDF |
| Text Splitting | LangChain |
| Embeddings | `qwen3-embedding:0.6b` via sentence-transformers / Ollama |
| Vector Store | Supabase (PostgreSQL + pgvector) |
| Inference | `llama3:8b` via Ollama |
| Frontend | Next.js 16, React 19, TypeScript, Tailwind CSS |

## Project Structure

```
├── ingest.py                # PDF ingestion pipeline
├── test_embeddings.py       # Test retrieval with sample queries
├── requirements.txt         # Python dependencies
├── human-nutrition-text.pdf # Source textbook
└── rag-chat/                # Next.js chat frontend
    └── src/app/
        ├── api/chat/route.ts   # RAG API endpoint
        └── page.tsx            # Chat UI
```

## Setup

### Prerequisites

- Python 3.10+
- Node.js 18+
- [Ollama](https://ollama.com) running locally
- A [Supabase](https://supabase.com) project with pgvector enabled and a `match_documents` RPC function

### 1. Pull Ollama models

```bash
ollama pull qwen3-embedding:0.6b
ollama pull llama3:8b
ollama serve
```

### 2. Configure environment variables

**Root `.env`**
```
SUPABASE_URL=<your-supabase-url>
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
```

**`rag-chat/.env.local`**
```
SUPABASE_URL=<your-supabase-url>
SUPABASE_SERVICE_ROLE_KEY=<your-service-role-key>
EMBEDDING_API_URL=
INFERENCE_API_URL=
```

### 3. Ingest the PDF

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python ingest.py
```

### 4. Run the chat app

```bash
cd rag-chat
npm install
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) and start asking nutrition questions.
