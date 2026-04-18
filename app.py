import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.utils import secure_filename

# Import our custom RAG modules
from ingestor import process_pdf
from retriever import create_vector_store
from chatbot import ask

# Load environment variables (e.g. GOOGLE_API_KEY)
load_dotenv()

app = Flask(__name__)
# Enable Cross-Origin Resource Sharing
CORS(app)

# Ensure the uploads directory exists relative to execution path
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def index():
    """Serves the frontend static/index.html interface."""
    return app.send_static_file('index.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint to verify API is active."""
    return jsonify({"status": "ok"}), 200

@app.route('/upload', methods=['POST'])
def upload_file():
    """
    Accepts a PDF upload, extracts text, generates embeddings, 
    and saves the vector index locally.
    """
    if 'file' not in request.files:
        return jsonify({"error": "No file parameter found in the request"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400
        
    # Validate PDF extension
    if file and file.filename.lower().endswith('.pdf'):
        try:
            # Secure file name and save to the uploads folder
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Step 1: Run the document extraction and chunking pipeline via ingestor.py
            print(f"Processing newly uploaded file: {filename}")
            chunks = process_pdf(file_path)
            
            if not chunks:
                return jsonify({"error": "Failed to extract text. File might be an empty PDF or invalid."}), 500
                
            # Step 2: Ingest the chunks into the embedding vector store via retriever.py
            vector_store = create_vector_store(chunks)
            if not vector_store:
                return jsonify({"error": "Vector store creation failed internally."}), 500
                
            return jsonify({
                "status": "success", 
                "message": "Document processed"
            }), 200
            
        except Exception as e:
            # Catch file read or embedding generation errors cleanly
            return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500
            
    else:
        return jsonify({"error": "Unsupported file format. Please upload a PDF."}), 400

@app.route('/ask', methods=['POST'])
def ask_question():
    """
    Accepts a JSON payload with a question and uses the RAG 
    pipeline to formulate an answer with source references.
    """
    try:
        data = request.get_json()
        if not data or 'question' not in data:
            return jsonify({"error": "Missing 'question' attribute in JSON request body"}), 400
            
        question = data['question']
        
        # Delegate QA retrieval to chatbot.py
        answer_text, source_pages = ask(question)
        
        # Format the output sources nicely as expected: ["Page 3", "Page 7"]
        formatted_sources = [f"Page {page}" for page in source_pages]
        
        return jsonify({
            "answer": answer_text,
            "sources": formatted_sources
        }), 200
        
    except FileNotFoundError:
        return jsonify({
            "error": "Backend index is missing. Please upload a document using the /upload endpoint first."
        }), 400
    except Exception as e:
        return jsonify({
            "error": f"An unhandled error occurred while querying: {str(e)}"
        }), 500

if __name__ == '__main__':
    # Launch the development server
    app.run(debug=True, host='0.0.0.0', port=5000)
