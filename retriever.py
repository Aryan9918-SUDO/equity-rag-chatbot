import os
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Load environment variables from .env file
load_dotenv()

INDEX_DIR = "faiss_index"

def _get_embeddings_model():
    """Initialize and return the Google Generative AI Embeddings model."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
        
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=api_key
    )

def create_vector_store(chunks):
    """
    Creates embeddings for the provided text chunks and stores them locally via FAISS.
    
    Args:
        chunks (list): A list of dictionaries containing 'page_content' and 'metadata'.
    """
    if not chunks:
        print("No chunks provided. Skipping vector store creation.")
        return None
        
    print(f"Creating vector store for {len(chunks)} chunks...")
    
    # Extract texts and metadatas for FAISS
    texts = [chunk.get("page_content", "") for chunk in chunks]
    metadatas = [chunk.get("metadata", {}) for chunk in chunks]
    
    embeddings = _get_embeddings_model()
    
    # Generate embeddings and populate FAISS vector store
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    
    # Save the index locally to the specified directory
    vector_store.save_local(INDEX_DIR)
    print(f"FAISS index successfully saved to the '{INDEX_DIR}' directory.")
    
    return vector_store

def load_retriever():
    """
    Loads the locally stored FAISS vector index and returns a retriever.
    
    Returns:
        A LangChain retriever object configured to fetch the top 4 relevant documents.
    """
    if not os.path.exists(INDEX_DIR):
        raise FileNotFoundError(f"FAISS index directory '{INDEX_DIR}' not found. You must create it first.")
        
    embeddings = _get_embeddings_model()
    
    # load_local requires allow_dangerous_deserialization=True in newer langchain versions
    # This is safe here because we generated these files locally ourselves.
    vector_store = FAISS.load_local(
        folder_path=INDEX_DIR, 
        embeddings=embeddings, 
        allow_dangerous_deserialization=True
    )
    
    # Return a retriever that fetches the top 4 chunks (k=4)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})
    return retriever

if __name__ == "__main__":
    # Optional test block to showcase how to use the modules independently
    # Note: testing requires a valid GOOGLE_API_KEY in the .env file.
    pass
