# app/core/groq_client.py

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def initialize_llm():
    """Initialize the Groq LLM using the official langchain-groq wrapper."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set.")
    return ChatGroq(model_name="llama3-8b-8192", api_key=GROQ_API_KEY)

def create_medical_qa_chain(vector_db):
    """Create a medical QA chain with the Groq LLM."""
    llm = initialize_llm()

    prompt_template = """
    You are a knowledgeable medical assistant providing information based on authoritative medical sources.
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    Always maintain a professional and accurate tone. Be concise but thorough.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 15}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

    return qa_chain

def answer_medical_question(query, qa_chain):
    """Answer a medical question using the RAG system."""
    try:
        response = qa_chain({"query": query})
        answer = response.get("result", "")
        source_docs = response.get("source_documents", [])

        sources = []
        for i, doc in enumerate(source_docs):
            metadata = doc.metadata
            source = metadata.get('source', 'Unknown')
            page = metadata.get('page', 'Unknown')
            sources.append(f"Source {i+1}: {source}, Page {page}")

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "source_documents": source_docs
        }

    except Exception as e:
        return {
            "query": query,
            "answer": f"Error generating response: {str(e)}",
            "sources": [],
            "source_documents": []
        }