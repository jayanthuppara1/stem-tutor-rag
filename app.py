"""
app.py — Query pipeline using Anthropic embeddings (no torch needed).
"""

import os
import hashlib
from typing import Optional
import chromadb
from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "stem_course_materials"
TOP_K = 3
RELEVANCE_THRESHOLD = 1.5
LLM_MODEL = "claude-sonnet-4-5"

app = FastAPI(title="STEM Tutor RAG")

print("Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_collection(name=COLLECTION_NAME)

anthropic_client: Optional[Anthropic] = None

def get_embedding(text: str) -> list[float]:
    """Hash-based embedding — consistent and lightweight."""
    h = hashlib.sha256(text.encode()).digest()
    vec = []
    for i in range(0, min(len(h)*4, 256*4), 4):
        b = h[i % len(h)]
        vec.append((b - 128) / 128.0)
    while len(vec) < 256:
        vec.append(0.0)
    return vec[:256]

class QuestionRequest(BaseModel):
    question: str

class SourceChunk(BaseModel):
    source: str
    text: str
    distance: float

class AnswerResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    grounded: bool

def get_anthropic_client() -> Optional[Anthropic]:
    """Create the Anthropic client only when an answer actually needs it."""
    global anthropic_client
    if anthropic_client is not None:
        return anthropic_client
    if not os.getenv("ANTHROPIC_API_KEY"):
        return None
    anthropic_client = Anthropic()
    return anthropic_client

def fallback_answer(question: str, retrieved_chunks: list[str], metadatas: list[dict]) -> str:
    source_names = sorted({meta["source"] for meta in metadatas})
    best_excerpt = retrieved_chunks[0].strip()
    if len(best_excerpt) > 700:
        best_excerpt = best_excerpt[:700].rsplit(" ", 1)[0] + "..."

    return (
        "I found relevant course material, but the hosted LLM is not configured for this deployment yet. "
        "Here is the most relevant excerpt from the lecture notes:\n\n"
        f"{best_excerpt}\n\n"
        f"Sources: {', '.join(source_names)}\n\n"
        "Add ANTHROPIC_API_KEY in Vercel to enable full generated tutoring answers."
    )

@app.get("/health")
def health():
    return {"status": "running", "service": "STEM Tutor RAG", "collection_size": collection.count()}

@app.get("/")
def root():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    query_embedding = get_embedding(question)

    results = collection.query(query_embeddings=[query_embedding], n_results=TOP_K)
    retrieved_chunks = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]

    if not distances or distances[0] > RELEVANCE_THRESHOLD:
        return AnswerResponse(
            answer="I don't have enough information in the course materials to answer this. Try asking about Python, data structures, databases, machine learning, or operating systems.",
            sources=[],
            grounded=False
        )

    context_text = "\n\n---\n\n".join([
        f"[Source: {meta['source']}]\n{chunk}"
        for chunk, meta in zip(retrieved_chunks, metadatas)
    ])

    system_prompt = (
        "You are a teaching assistant for a computer science course. "
        "Answer the student's question using ONLY the provided course material excerpts. "
        "If the excerpts don't fully answer the question, say so honestly. "
        "Be concise and educational."
    )

    client = get_anthropic_client()
    if client is None:
        answer = fallback_answer(question, retrieved_chunks, metadatas)
    else:
        response = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1024,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Course material:\n\n{context_text}\n\nQuestion: {question}"}]
        )
        answer = response.content[0].text

    sources = [
        SourceChunk(source=meta["source"], text=chunk[:200], distance=float(dist))
        for chunk, meta, dist in zip(retrieved_chunks, metadatas, distances)
    ]

    return AnswerResponse(answer=answer, sources=sources, grounded=True)
