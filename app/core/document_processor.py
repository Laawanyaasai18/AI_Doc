# app/core/document_processor.py
import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Define paths
DATA_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"
DB_CHROMA_PATH = "vectorstore/db_chroma"

# Step 1: Load Raw PDF(s)
def load_pdf_files(data_path=DATA_PATH):
    """
    Load all PDF files from a directory and extract text.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Error: Directory '{data_path}' does not exist.")
    
    loader = DirectoryLoader(data_path, glob="*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        print("Warning: No PDFs found in the directory.")
    
    return documents

# Step 2: Chunk the Text
def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    """
    Splits documents into smaller chunks for efficient embedding.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents)

# Step 3: Create Vector Embeddings
def get_embeddings():
    """
    Loads a Hugging Face model for embedding creation.
    """
    print("✅ Step 3: Generating embeddings...")
    return HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Step 4: Store Embeddings in ChromaDB
def store_embeddings_chroma(chunks, embed_model):
    """
    Creates ChromaDB database for storing and retrieving embeddings.
    """
    if not chunks:
        print("⚠️ Warning: No chunks found. Skipping ChromaDB storage.")
        return None
    
    print("✅ Step 4: Creating ChromaDB database...")
    
    # Create directory if it doesn't exist
    os.makedirs(DB_CHROMA_PATH, exist_ok=True)
    
    # Create and persist the database
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embed_model,
        persist_directory=DB_CHROMA_PATH
    )
    db.persist()
    
    print("✅ ChromaDB database saved successfully!")
    return db