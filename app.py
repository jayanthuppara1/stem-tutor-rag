"""
app.py — Query pipeline for the STEM Tutor RAG system.

This FastAPI server handles student questions:
1. Receives a question via POST /ask
2. Embeds the question using the same model used in ingestion
3. Retrieves the top-3 most similar chunks from ChromaDB
4. Checks relevance — if best match is too weak, refuses to answer
5. Builds a prompt with retrieved context
6. Calls Claude to generate a grounded answer
7. Returns the answer plus the source chunks used

Run with: uvicorn app:app --reload
"""

import os
from typing import Optional

import chromadb
from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer


# Configuration — must match ingest.py
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "stem_course_materials"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 3                       # number of chunks to retrieve
RELEVANCE_THRESHOLD = 1.8       # max distance to consider relevant (lower = more similar)
LLM_MODEL = "claude-sonnet-4-5"


# Initialize FastAPI app
app = FastAPI(
    title="STEM Tutor RAG",
    description="AI-powered Q&A grounded in actual course materials",
    version="1.0.0"
)

# Load components once at startup (faster than per-request)
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

print("Initializing Anthropic client...")
anthropic_client = Anthropic()  # reads ANTHROPIC_API_KEY from environment


# Request/response data models
class QuestionRequest(BaseModel):
    question: str


class SourceChunk(BaseModel):
    source: str
    chunk_index: int
    text: str
    distance: float


class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    grounded: bool  # whether the answer used retrieved context


@app.get("/health")
def health():
    """Health check endpoint."""
    return {
        "status": "running",
        "service": "STEM Tutor RAG",
        "collection_size": collection.count()
    }

@app.get("/")
def root():
    """Serve the frontend UI."""
    return FileResponse("static/index.html")

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    """
    Answer a student's question using retrieval-augmented generation.
    """
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    # Step 1: Embed the question using the same model as ingestion
    # (critical — query and document embeddings must come from the same model)
    query_embedding = embedding_model.encode(question).tolist()

    # Step 2: Retrieve top-K most similar chunks from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K
    )

    # ChromaDB returns nested lists (one per query); we only sent one query
    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    # Step 3: Relevance check — if even the best match is too weak,
    # refuse to answer rather than hallucinate
    if not distances or distances[0] > RELEVANCE_THRESHOLD:
        return AnswerResponse(
            answer=(
                "I don't have enough information in the course materials "
                "to answer this question. Try rephrasing or asking about "
                "topics covered in the lectures (Python, data structures, "
                "databases, machine learning, or operating systems)."
            ),
            sources=[],
            grounded=False
        )

    # Step 4: Build the prompt — system instruction + retrieved context + question
    context_text = "\n\n---\n\n".join([
        f"[Source: {meta['source']}]\n{chunk}"
        for chunk, meta in zip(retrieved_chunks, metadatas)
    ])

    system_prompt = (
        "You are a teaching assistant for a computer science course. "
        "Answer the student's question using ONLY the provided course material excerpts. "
        "If the excerpts don't fully answer the question, say so honestly. "
        "Cite the source filename when relevant. Be concise and educational."
    )

    user_message = (
        f"Course material excerpts:\n\n{context_text}\n\n"
        f"Student question: {question}"
    )

    # Step 5: Call Claude to generate the answer
    response = anthropic_client.messages.create(
        model=LLM_MODEL,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}]
    )

    answer_text = response.content[0].text

    # Step 6: Build the response with sources for transparency
    sources = [
        SourceChunk(
            source=meta["source"],
            chunk_index=meta["chunk_index"],
            text=chunk[:200] + ("..." if len(chunk) > 200 else ""),
            distance=float(dist)
        )
        for chunk, meta, dist in zip(retrieved_chunks, metadatas, distances)
    ]

    return AnswerResponse(
        answer=answer_text,
        sources=sources,
        grounded=True
    )
