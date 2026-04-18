import os
import fitz  # PyMuPDF
from datetime import datetime
from pymongo import MongoClient
import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

def process_pdf(file_path, chunk_size=500, chunk_overlap=50):
    """
    Accepts a PDF file path, extracts text page by page, 
    splits it into chunks, and returns them with metadata.
    """
    filename = os.path.basename(file_path)
    chunks_with_metadata = []
    
    # Initialize MongoDB Connection
    mongo_uri = os.getenv("MONGODB_URI")
    mongo_collection = None
    if mongo_uri:
        try:
            client = MongoClient(
                mongo_uri,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
                tlsCAFile=certifi.where()
            )
            db = client["equity_rag"]
            mongo_collection = db["document_chunks"]
            
            # Delete any existing chunks with the same source filename
            mongo_collection.delete_many({"source": filename})
            print(f"Cleared existing MongoDB chunks for {filename}")
        except Exception as e:
            print(f"MongoDB connection failed: {e}")
            mongo_collection = None
    else:
        print("MONGODB_URI not found in .env. Skipping MongoDB insertion.")
    
    # Initialize LangChain's RecursiveCharacterTextSplitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    
    try:
        # Open the PDF file
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Error opening PDF {file_path}: {e}")
        return []

    # Iterate through each page
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        
        if not text.strip():
            continue  # Skip empty pages
            
        # Split the text of the current page into chunks
        page_chunks = text_splitter.split_text(text)
        
        # Attach metadata to each chunk
        for chunk_idx, chunk in enumerate(page_chunks):
            # 1. LangChain format for FAISS
            chunk_data = {
                "page_content": chunk,
                "metadata": {
                    "source": filename,
                    "page": page_num + 1  # Using 1-indexed page numbers
                }
            }
            chunks_with_metadata.append(chunk_data)
            
            # 2. MongoDB format for source metadata tracking
            if mongo_collection is not None:
                mongo_doc = {
                    "text": chunk,
                    "page_number": page_num + 1,
                    "source": filename,
                    "chunk_index": chunk_idx,
                    "timestamp": datetime.utcnow()
                }
                try:
                    mongo_collection.insert_one(mongo_doc)
                except Exception as e:
                    print(f"Failed to insert chunk into MongoDB: {e}")
            
    doc.close()
    return chunks_with_metadata

if __name__ == "__main__":
    # Test block
    sample_pdf_path = "sample_test.pdf"
    
    # Create a simple dummy PDF using PyMuPDF to test if it doesn't already exist
    if not os.path.exists(sample_pdf_path):
        dummy_doc = fitz.open()
        page = dummy_doc.new_page()
        dummy_text = "This is a sample sentence to test the ingestor extracting and chunking logic. " * 30
        page.insert_textbox(fitz.Rect(50, 50, 500, 700), dummy_text)
        dummy_doc.save(sample_pdf_path)
        dummy_doc.close()
        print(f"Created dummy PDF for testing at: {sample_pdf_path}\n")

    print(f"Running ingestor on: {sample_pdf_path}")
    extracted_chunks = process_pdf(sample_pdf_path)
    
    print(f"Total chunks extracted: {len(extracted_chunks)}\n")
    print("--- First 3 Chunks ---")
    
    for i, chunk in enumerate(extracted_chunks[:3]):
        print(f"Chunk {i+1}:")
        print(f"Metadata: {chunk['metadata']}")
        print(f"Content: {chunk['page_content']}")
        print("-" * 40)
