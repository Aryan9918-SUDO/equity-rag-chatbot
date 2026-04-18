# 📄 Equity Document Q&A Chatbot

> An intelligent document automation tool powered by **RAG**, **LangChain**, and **Google Gemini** to effortlessly extract insights from complex equity contracts and ESOP policies.

---

## 🚀 What it does

This tool uses a technique called **Retrieval-Augmented Generation (RAG)**. In simple terms, RAG allows an AI to read through a massive custom document (like a complex legal PDF) and retrieve the exact paragraphs related to your question *before* attempting to formulate an answer. This directly grounds the AI in reality and prevents hallucinations. 

By employing this system, the Equity Document Q&A Chatbot functions similarly to **Qapita’s document intelligence system**—dynamically answering specific questions regarding vesting schedules, cliff periods, and eligibility criteria while explicitly citing the exact pages those answers were drawn from.

---

## 🛠️ Tech Stack

- **Python** (Core backend logic)
- **LangChain** (Framework for robust LLM orchestration)
- **Google Gemini** (Advanced LLM & Vector Embeddings backend)
- **FAISS** (Facebook AI Similarity Search library for local vector indexing)
- **PyMuPDF / fitz** (High-speed PDF text and format extraction)
- **Flask** (Lightweight backend REST API)
- **MongoDB** (Extensible database framework for overarching document/tenant metadata tracking)

---

## 🏗️ Architecture Pipeline

This application processes and parses complex legal PDFs automatically through a scalable and robust pipeline:

1. **PDF Upload**: The user uploads an ESOP or Equity PDF securely via the minimalist frontend.
2. **Text Extraction**: The `ingestor.py` script parses the file iteratively page-by-page.
3. **Chunking**: The extracted text is dynamically broken into continuous 500-character chunks utilizing precise semantic overlaps.
4. **Embedding**: Chunks are passed to Google's specialized text-embedding models to convert human text into mathematical vector arrays.
5. **Vector Store**: The embeddings are intelligently persisted locally within a high-speed FAISS database.
6. **Retrieval**: When a query is asked via the UI, the system matches the mathematical intent of the question to the nearest document chunks.
7. **LLM Synthesis**: The exact semantic matches are injected into the Gemini LLM alongside restrictive prompting to return a concise, factual answer complemented uniformly by **direct source page citations**.

---

## ⚙️ How to run

### 1. Requirements and Installation
Ensure you have Python 3 installed. Navigate to the root directory of the software.

```bash
# Create and activate an isolated virtual environment
python3 -m venv venv
source venv/bin/activate

# Install all mandatory dependencies
pip install langchain langchain-community langchain-google-genai pymupdf flask flask-cors pymongo python-dotenv faiss-cpu tiktoken
```

### 2. Environment Setup
Create a `.env` file in the root directory and securely configure your Google API Key:

```ini
GOOGLE_API_KEY=your_gemini_api_key_here
MONGODB_URI=your_mongo_db_uri_here
```

### 3. Start the Local Server
Launch the Flask backend process using the provided standard entry point:

```bash
python3 app.py
```

### 4. Access the Application
Open your preferred web browser and navigate directly to:
```
http://127.0.0.1:5000
```
Simply upload your PDF via the clean frontend interface and begin querying! 

*(Optional: For programmatic pipeline testing, execute `python3 test.py` to watch the ingestion pipeline and RAG chain run predefined legal queries automatically.)*

---

> **Note**: *This project was heavily inspired by modern equity management platforms utilizing Retrieval-Augmented Generation (RAG) paradigms for sophisticated document intelligence and large scale legal analysis.*
