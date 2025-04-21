# app/core/retrieval.py
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Path to the ChromaDB
DB_CHROMA_PATH = "vectorstore/db_chroma"

def load_embeddings():
    """Load the embedding model."""
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

def load_vectorstore():
    """Load the ChromaDB vector store."""
    if not os.path.exists(DB_CHROMA_PATH):
        raise FileNotFoundError(f"Error: ChromaDB at '{DB_CHROMA_PATH}' does not exist. Process documents first.")
    
    # Load the embedding function
    embeddings = load_embeddings()
    
    # Load the persisted database
    db = Chroma(persist_directory=DB_CHROMA_PATH, embedding_function=embeddings)
    print(f"✅ Loaded ChromaDB from {DB_CHROMA_PATH}")
    
    return db

def similarity_search(query, db=None, k=5):
    """
    Search for documents similar to the query.
    """
    # Load the database if not provided
    if db is None:
        db = load_vectorstore()
    
    # Perform similarity search
    results = db.similarity_search(query, k=k)
    return results

def mmr_search(query, db=None, k=5, diversity=0.5):
    """
    Search for documents using Maximum Marginal Relevance to ensure diversity.
    """
    # Load the database if not provided
    if db is None:
        db = load_vectorstore()
    
    # Perform MMR search
    results = db.max_marginal_relevance_search(query, k=k, fetch_k=15, lambda_mult=diversity)
    return results

def hybrid_search(query, db=None, k=5):
    """
    Hybrid search using both similarity and MMR for better results.
    """
    # Get regular similarity results
    sim_results = similarity_search(query, db, k=k)
    
    # Get diverse results with MMR
    mmr_results = mmr_search(query, db, k=k)
    
    # Combine and deduplicate results
    seen_content = set()
    combined_results = []
    
    # Add similarity results first
    for doc in sim_results:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            combined_results.append(doc)
    
    # Add MMR results that aren't duplicates
    for doc in mmr_results:
        if doc.page_content not in seen_content and len(combined_results) < k:
            seen_content.add(doc.page_content)
            combined_results.append(doc)
    
    return combined_results[:k]

def format_results(results):
    """
    Format search results for display.
    """
    formatted_output = []
    for i, doc in enumerate(results):
        content = doc.page_content
        metadata = doc.metadata
        
        # Extract source information
        source = metadata.get('source', 'Unknown')
        page = metadata.get('page', 'Unknown')
        
        # Format the result
        result = {
            "id": i,
            "content": content,
            "source": source,
            "page": page,
            "metadata": metadata
        }
        
        formatted_output.append(result)
    
    return formatted_output