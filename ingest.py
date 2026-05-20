import os
import hashlib
from pathlib import Path
import chromadb

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
NOTES_DIR = "lecture_notes"
CHROMA_DIR = "chroma_db"
COLLECTION_NAME = "stem_course_materials"

def get_embedding(text: str) -> list[float]:
    h = hashlib.sha256(text.encode()).digest()
    vec = []
    for i in range(256):
        b = h[i % len(h)]
        vec.append((b - 128) / 128.0)
    return vec

def load_lecture_files(directory: str) -> list[dict]:
    notes = []
    for file_path in sorted(Path(directory).glob("*.txt")):
        with open(file_path, "r", encoding="utf-8") as f:
            notes.append({"filename": file_path.name, "content": f.read()})
    print(f"Loaded {len(notes)} lecture files")
    return notes

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

def main():
    print("STEM Tutor RAG — Ingestion Pipeline")
    notes = load_lecture_files(NOTES_DIR)
    chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
    except Exception:
        pass
    collection = chroma_client.create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})
    total_chunks = 0
    for note in notes:
        chunks = chunk_text(note["content"])
        print(f"{note['filename']}: {len(chunks)} chunks")
        embeddings = [get_embedding(chunk) for chunk in chunks]
        ids = [f"{note['filename']}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [{"source": note["filename"], "chunk_index": i} for i in range(len(chunks))]
        collection.add(documents=chunks, embeddings=embeddings, ids=ids, metadatas=metadatas)
        total_chunks += len(chunks)
    print(f"Done: {total_chunks} chunks stored")

if __name__ == "__main__":
    main()
