# STEM Tutor RAG

An AI-powered Q&A chatbot that answers questions grounded in actual course materials, not hallucinated responses.

## What this is

A retrieval-augmented generation (RAG) system that lets students ask questions about CS course topics and get answers backed by real lecture notes — with cited sources.

I originally built a system like this during my graduate work at USF for STEM courses. That code lived on university infrastructure and stayed there when I graduated. This is a clean rebuild with modern tooling.

## Architecture

```
Lecture notes (.txt files)
  ↓
[Chunking: 500 chars, 50 overlap, sentence boundaries]
  ↓
[Embedding: all-MiniLM-L6-v2 → 384-dim vectors]
  ↓
[ChromaDB persistent store]


Student question
  ↓
[Same embedding model → query vector]
  ↓
[ChromaDB similarity search → top 3 chunks]
  ↓
[Relevance check — refuse if no good match]
  ↓
[Claude with retrieved context as the system prompt]
  ↓
Grounded answer + cited sources
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI | Async, modern, automatic docs |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) | Runs locally, no API cost, fast |
| Vector DB | ChromaDB | Easy local persistence, good for portfolio scale |
| LLM | Claude 3.5 Sonnet via Anthropic API | Strong instruction following, refuses to hallucinate |

## Key design decisions

**Chunk size 500 chars with 50 overlap, breaking at sentences.**
Tradeoff: small enough for precise retrieval, large enough to carry context. Overlap prevents losing concepts that span a boundary. Sentence boundaries preserve semantic coherence.

**Relevance threshold.**
If the best matching chunk is too far from the query, the system refuses to answer rather than fabricate. This is the core hallucination defense.

**Source citations returned with every answer.**
Students can verify the answer against the original material. Trust is critical in an educational context.

## Setup

```bash
# Clone and enter the project
git clone <your-repo-url>
cd stem-tutor-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Anthropic API key
export ANTHROPIC_API_KEY=sk-ant-...

# Run ingestion (one time)
python ingest.py

# Start the server
uvicorn app:app --reload
```

Then open http://localhost:8000/docs to test in your browser.

## Example query

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the difference between a list and a tuple in Python?"}'
```

## What I learned rebuilding this

- The embedding model matters more than the LLM for retrieval quality. Switching from a smaller to a more capable embedding model improved retrieval far more than swapping LLMs.
- Sentence-boundary chunking beat fixed-size chunking in my testing — the chunks contained more complete thoughts.
- The relevance threshold needs tuning per dataset. Too tight and the bot refuses good questions; too loose and it hallucinates.

## Future improvements

- Hybrid search (BM25 + vector) for better recall on keyword-heavy queries
- Reranking step using a cross-encoder
- Query rewriting to handle vague student questions
- Streaming responses for better UX
