import os
from pathlib import Path
from pypdf import PdfReader
from src.utils.vector_db import get_qdrant_client

def ingest_pdfs(data_dir: str = "data", collection_name: str = "phq9_docs"):
    """
    Reads PDFs from data_dir, chunks them, and upserts to Qdrant.
    """
    client = get_qdrant_client()
    data_path = Path.home() / "emerald-oort" / data_dir
    
    if not data_path.exists():
        print(f"Data directory {data_path} does not exist.")
        return

    docs = []
    ids = []
    
    print(f"Scanning {data_path} for PDFs...")
    
    for file in data_path.glob("*.pdf"):
        print(f"Processing {file.name}...")
        try:
            reader = PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            # Simple chunking by paragraphs or size could be added here.
            # For now, we ingest the whole document or large chunks if needed.
            # Let's do simple 500 char overlapping chunks for better retrieval.
            
            chunk_size = 1000
            overlap = 100
            
            start = 0
            chunk_idx = 0
            while start < len(text):
                end = start + chunk_size
                chunk = text[start:end]
                docs.append(chunk)
                
                # Deterministic ID based on filename and chunk index
                # Using simple integers or strings. Qdrant likes UUIDs or ints.
                # Let's use simple string IDs or let Qdrant auto-generate (if we pass None)
                # But client.add expects lists.
                
                # Deterministic UUID based on filename and chunk index
                import uuid
                namespace = uuid.UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8') # DNS namespace as example
                ids.append(str(uuid.uuid5(namespace, f"{file.name}_{chunk_idx}")))
                
                start += (chunk_size - overlap)
                chunk_idx += 1
                
        except Exception as e:
            print(f"Error processing {file.name}: {e}")

    if not docs:
        print("No documents found to ingest.")
        return

    print(f"Upserting {len(docs)} chunks to collection '{collection_name}'...")
    
    # client.add automatically handles embedding generation with FastEmbed
    client.add(
        collection_name=collection_name,
        documents=docs,
        ids=ids
    )
    
    print("Ingestion complete.")

if __name__ == "__main__":
    ingest_pdfs()
