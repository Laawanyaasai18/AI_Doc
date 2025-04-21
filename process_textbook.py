# process_textbook.py
import os
from app.core.document_processor import (
    load_pdf_files, 
    chunk_documents, 
    get_embeddings, 
    store_embeddings_chroma,
    DATA_PATH
)

if __name__ == "__main__":
    print("🔄 Starting medical textbook processing pipeline...")
    
    # Step 1: Load PDFs
    print("📚 Loading PDFs...")
    docs = load_pdf_files(DATA_PATH)
    print(f"✅ Loaded {len(docs)} documents from PDFs.")
    
    # Step 2: Chunk documents
    print("✂️ Chunking documents...")
    chunked_docs = chunk_documents(docs)
    print(f"✅ Created {len(chunked_docs)} text chunks.")
    
    # Step 3: Generate embeddings
    print("🧠 Generating embeddings...")
    embed_model = get_embeddings()
    
    # Step 4: Store in ChromaDB
    chroma_db = store_embeddings_chroma(chunked_docs, embed_model)
    
    if chroma_db:
        print("🚀 ChromaDB embedding storage process completed!")
    else:
        print("⚠️ ChromaDB storage skipped due to missing chunks.")
        