import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from retriever import load_retriever

# Load environment variables
load_dotenv()

# Define the custom prompt as requested
prompt_template = """
You are an intelligent assistant. Answer the question based ONLY on the provided context below.
If you cannot find the answer in the context, exactly say: "I cannot find this information in the document".
Always mention which page the answer came from in your response based on the context provided.

Context: {context}

Question: {question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

# Holds the singleton instance of our RetrievalQA chain
_qa_chain = None

def _get_qa_chain():
    """Initializes and returns the RetrievalQA chain lazily."""
    global _qa_chain
    if _qa_chain is None:
        # Load the retriever from our retriever module
        retriever = load_retriever()
        
        # Initialize Google GenAI chat model
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found. Please set it in your .env file.")
            
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash", 
            google_api_key=api_key,
            temperature=0  # Set lowest temperature for strict factual retrieval
        )
        
        # Create the RAG chain using the "stuff" document strategy
        _qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    return _qa_chain

def ask(question):
    """
    Submits a question to the RAG pipeline.
    
    Returns:
        tuple: (answer_text, source_pages)
    """
    chain = _get_qa_chain()
    
    # Run the chain using the new invoke format backward compatible with RetrievalQA
    response = chain.invoke({"query": question})
    
    answer_text = response['result']
    
    # If the model explicitly couldn't find the answer in the text, clear the misleading sources
    lower_answer = answer_text.lower()
    if "cannot find this information" in lower_answer or "not in the document" in lower_answer:
        return answer_text, []
        
    source_docs = response.get('source_documents', [])
    
    # Extract robust, unique page numbers from the fetched documents
    source_pages = list(set([doc.metadata.get("page", "Unknown") for doc in source_docs]))
    
    # Format and return the tuple
    return answer_text, sorted(source_pages)

if __name__ == "__main__":
    # Test block
    test_question = "What is the vesting period?"
    print(f"Testing the chatbot module...")
    print(f"Question: '{test_question}'\n")
    
    try:
        answer, pages = ask(test_question)
        print("--- Response ---")
        print(f"Answer: {answer}")
        print(f"Source Pages: {pages}")
    except FileNotFoundError as e:
        print(f"Module Failed: {e}")
        print("Have you generated your local FAISS index yet? You must run the vector store creation first before querying!")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
