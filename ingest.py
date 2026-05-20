"""
ingest.py — Ingestion pipeline for the STEM Tutor RAG system.

This script does the offline work of preparing course materials for retrieval:
1. Loads lecture notes from text files
2. Splits them into chunks (500 chars, 50 char overlap)
3. Generates embeddings using sentence-transformers (all-MiniLM-L6-v2)
4. Stores chunks + embeddings in ChromaDB for fast similarity search

Run this once before starting the server.
"""

import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Configuration — these match best practices from the course
CHUNK_SIZE = 500          # characters per chunk
CHUNK_OVERLAP = 50        # overlap between adjacent chunks
NOTES_DIR = "lecture_notes"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "stem_course_materials"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_lecture_files(directory: str) -> list[dict]:
    """
    Read all .txt files in the notes directory.
    Returns a list of dicts with 'filename' and 'content'.
    """
    notes = []
    notes_path = Path(directory)
    if not notes_path.exists():
        raise FileNotFoundError(f"Directory '{directory}' not found")

    for file_path in sorted(notes_path.glob("*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            notes.append({
                "filename": file_path.name,
                "content": f.read()
            })
    print(f"Loaded {len(notes)} lecture files")
    return notes


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks at sentence boundaries when possible.

    Why overlap? If a key concept spans the chunk boundary, overlap ensures
    at least one chunk contains it fully — so retrieval doesn't miss it.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence end (period followed by space)
        # to avoid cutting mid-sentence — better semantic coherence
        if end < len(text):
            # Look for sentence boundary in the last 100 chars of the chunk
            sentence_end = text.rfind(". ", start, end)
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 2  # include the period and space

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move forward by chunk_size minus overlap
        start = end - overlap if end < len(text) else end

    return chunks


def main():
    print("=" * 60)
    print("STEM Tutor RAG — Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Load all lecture files
    notes = load_lecture_files(NOTES_DIR)

    # Step 2: Load the embedding model
    # all-MiniLM-L6-v2 is a 22.7M parameter model that converts text to
    # 384-dimensional vectors representing semantic meaning.
    # It runs locally (no API key needed) and is fast on CPU.
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Step 3: Initialize ChromaDB
    # PersistentClient saves to disk so data survives restarts.
    print(f"Initializing ChromaDB at ./{CHROMA_DIR}")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete old collection if it exists, so we always start fresh
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}'")
    except Exception:
        pass

    collection = client.create_collection(name=COLLECTION_NAME)

    # Step 4: Chunk each file and add to ChromaDB
    total_chunks = 0
    for note in notes:
        chunks = chunk_text(note["content"])
        print(f"\n{note['filename']}: {len(chunks)} chunks")

        # Generate embeddings for all chunks in this file at once (faster)
        embeddings = model.encode(chunks).tolist()

        # Create unique IDs and metadata for each chunk
        ids = [f"{note['filename']}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": note["filename"], "chunk_index": i}
            for i in range(len(chunks))
        ]

        # Add to ChromaDB — stores text, vector, ID, and metadata together
        collection.add(
            documents=chunks,
            embeddings=embeddings,
            ids=ids,
            metadatas=metadatas
        )
        total_chunks += len(chunks)

    print("\n" + "=" * 60)
    print(f"Ingestion complete: {total_chunks} chunks stored")
    print(f"Collection '{COLLECTION_NAME}' is ready for queries")
    print("=" * 60)


if __name__ == "__main__":
    main()
