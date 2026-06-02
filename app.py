"""
app.py — Query pipeline using Anthropic embeddings (no torch needed).
"""

import os
import re
from pathlib import Path
from typing import Optional
from anthropic import Anthropic
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

NOTES_DIR = Path("lecture_notes")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3
MIN_OVERLAP_SCORE = 0.04
LLM_MODEL = "claude-sonnet-4-5"

app = FastAPI(title="STEM Tutor RAG")

anthropic_client: Optional[Anthropic] = None

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

def chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            sentence_end = text.rfind(". ", start, end)
            if sentence_end > start + CHUNK_SIZE // 2:
                end = sentence_end + 2
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP if end < len(text) else end
    return chunks

def load_course_chunks() -> list[dict]:
    course_chunks = []
    for file_path in sorted(NOTES_DIR.glob("*.txt")):
        text = file_path.read_text(encoding="utf-8")
        for index, chunk in enumerate(chunk_text(text)):
            course_chunks.append({"source": file_path.name, "index": index, "text": chunk})
    return course_chunks

COURSE_CHUNKS = load_course_chunks()

def tokenize(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 2
    }

def retrieve_chunks(question: str) -> list[dict]:
    query_tokens = tokenize(question)
    if not query_tokens:
        return []

    ranked = []
    for chunk in COURSE_CHUNKS:
        chunk_tokens = tokenize(chunk["text"])
        overlap = query_tokens.intersection(chunk_tokens)
        if not overlap:
            continue
        score = len(overlap) / len(query_tokens)
        ranked.append({**chunk, "score": score, "distance": 1 - score})

    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked[:TOP_K]

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
    return {"status": "running", "service": "STEM Tutor RAG", "collection_size": len(COURSE_CHUNKS)}

@app.get("/")
def root():
    return FileResponse("static/index.html")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    matches = retrieve_chunks(question)

    if not matches or matches[0]["score"] < MIN_OVERLAP_SCORE:
        return AnswerResponse(
            answer="I don't have enough information in the course materials to answer this. Try asking about Python, data structures, databases, machine learning, or operating systems.",
            sources=[],
            grounded=False
        )

    retrieved_chunks = [match["text"] for match in matches]
    metadatas = [{"source": match["source"], "chunk_index": match["index"]} for match in matches]

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
        SourceChunk(source=match["source"], text=match["text"][:200], distance=float(match["distance"]))
        for match in matches
    ]

    return AnswerResponse(answer=answer, sources=sources, grounded=True)
