from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import google.generativeai as genai
import faiss
import numpy as np
import json
import os
import PyPDF2
import csv
from io import StringIO
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configure Gemini API
API_KEY = "AIzaSyBIkZTm-SVuIpeNUtpE1w_uvYLTZF5oOZw"
genai.configure(api_key=API_KEY)

# Global variables
documents = []
index = None
embedding_model = None

def initialize_ai():
    """Initialize the Gemini embedding model"""
    global embedding_model
    try:
        embedding_model = genai.GenerativeModel('gemini-pro')
        logger.info("Gemini AI initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize Gemini AI: {e}")
        raise

def get_embedding(text):
    """Get embedding for text using Gemini"""
    try:
        # For this demo, we'll use a simple approach
        # In production, you'd use proper embedding models
        result = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_document"
        )
        return np.array(result['embedding'], dtype=np.float32)
    except Exception as e:
        logger.error(f"Error getting embedding: {e}")
        # Fallback to random embedding for demo
        return np.random.rand(768).astype(np.float32)

def load_documents():
    """Load documents from JSON file"""
    global documents
    try:
        if os.path.exists('diabetes_abstracts.json'):
            with open('diabetes_abstracts.json', 'r', encoding='utf-8') as f:
                documents = json.load(f)
            logger.info(f"Loaded {len(documents)} documents")
        else:
            documents = []
            logger.info("No existing documents found, starting with empty collection")
    except Exception as e:
        logger.error(f"Error loading documents: {e}")
        documents = []

def save_documents():
    """Save documents to JSON file"""
    try:
        with open('diabetes_abstracts.json', 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(documents)} documents")
    except Exception as e:
        logger.error(f"Error saving documents: {e}")

def build_index():
    """Build FAISS index from documents"""
    global index
    if not documents:
        logger.info("No documents to index")
        return
    
    try:
        # Get embeddings for all documents
        embeddings = []
        for doc in documents:
            embedding = get_embedding(doc['text'])
            embeddings.append(embedding)
        
        # Create FAISS index
        embeddings_array = np.array(embeddings)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings_array)
        index.add(embeddings_array)
        
        # Save index
        faiss.write_index(index, 'faiss_index.bin')
        logger.info(f"Built FAISS index with {len(documents)} documents")
    except Exception as e:
        logger.error(f"Error building index: {e}")

def load_index():
    """Load FAISS index from file"""
    global index
    try:
        if os.path.exists('faiss_index.bin'):
            index = faiss.read_index('faiss_index.bin')
            logger.info("Loaded FAISS index")
        else:
            logger.info("No existing index found")
    except Exception as e:
        logger.error(f"Error loading index: {e}")

def search_documents(query, top_k=3):
    """Search documents using FAISS"""
    if not index or not documents:
        return []
    
    try:
        # Get query embedding
        query_embedding = get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx < len(documents):
                doc = documents[idx]
                results.append({
                    'document': doc,
                    'score': float(score),
                    'snippet': doc['text'][:200] + '...' if len(doc['text']) > 200 else doc['text']
                })
        
        return results
    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        return []

def generate_answer(query, search_results):
    """Generate answer using Gemini"""
    try:
        # Prepare context from search results
        context = "\n\n".join([
            f"Source: {result['document']['title']}\nContent: {result['document']['text']}"
            for result in search_results[:3]
        ])
        
        prompt = f"""Based on the following medical sources, provide a concise, patient-friendly answer (2-3 sentences) to the question: "{query}"

Medical Sources:
{context}

Instructions:
- Provide evidence-based information only
- Keep the answer concise (2-3 sentences)
- Use patient-friendly language
- If the sources don't contain relevant information, say so

Answer:"""

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        return "I apologize, but I'm unable to generate an answer at this time. Please try again later."

@app.route('/')
def index_page():
    """Serve the main page"""
    return render_template_string(open('index.html', 'r', encoding='utf-8').read())

@app.route('/api/search', methods=['POST'])
def search():
    """Search endpoint"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({'error': 'Query is required'}), 400
        
        # Search documents
        search_results = search_documents(query)
        
        # Generate answer
        answer = generate_answer(query, search_results)
        
        return jsonify({
            'answer': answer,
            'sources': search_results
        })
    except Exception as e:
        logger.error(f"Error in search endpoint: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """Upload and process files"""
    try:
        files = request.files.getlist('files')
        title = request.form.get('title', '')
        citation = request.form.get('citation', '')
        
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        processed_docs = []
        
        for file in files:
            if file.filename == '':
                continue
                
            # Create uploads directory if it doesn't exist
            os.makedirs('uploads', exist_ok=True)
            
            # Save file
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)
            
            # Extract text based on file type
            text = ""
            if file.filename.lower().endswith('.pdf'):
                text = extract_pdf_text(file_path)
            elif file.filename.lower().endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
            elif file.filename.lower().endswith('.csv'):
                text = extract_csv_text(file_path)
            
            if text:
                # Summarize text
                summarized_text = summarize_medical_text(text)
                
                # Create document
                doc = {
                    'id': f"doc_{len(documents) + len(processed_docs) + 1}",
                    'title': title or file.filename,
                    'text': summarized_text,
                    'citation': citation
                }
                processed_docs.append(doc)
        
        # Add to documents and rebuild index
        documents.extend(processed_docs)
        save_documents()
        build_index()
        
        return jsonify({
            'message': f'Successfully processed {len(processed_docs)} documents',
            'documents': processed_docs
        })
    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        return jsonify({'error': 'Failed to process files'}), 500

@app.route('/api/add_text', methods=['POST'])
def add_text():
    """Add text content"""
    try:
        data = request.get_json()
        title = data.get('title', '')
        text = data.get('text', '')
        citation = data.get('citation', '')
        
        if not title or not text:
            return jsonify({'error': 'Title and text are required'}), 400
        
        # Summarize text
        summarized_text = summarize_medical_text(text)
        
        # Create document
        doc = {
            'id': f"doc_{len(documents) + 1}",
            'title': title,
            'text': summarized_text,
            'citation': citation
        }
        
        # Add to documents and rebuild index
        documents.append(doc)
        save_documents()
        build_index()
        
        return jsonify({
            'message': 'Successfully added text content',
            'document': doc
        })
    except Exception as e:
        logger.error(f"Error adding text: {e}")
        return jsonify({'error': 'Failed to add text content'}), 500

def extract_pdf_text(file_path):
    """Extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        logger.error(f"Error extracting PDF text: {e}")
        return ""

def extract_csv_text(file_path):
    """Extract text from CSV file"""
    try:
        text = ""
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                text += " ".join(row) + "\n"
        return text
    except Exception as e:
        logger.error(f"Error extracting CSV text: {e}")
        return ""

def summarize_medical_text(text):
    """Summarize medical text using Gemini"""
    try:
        prompt = f"""Summarize the following medical text, focusing on key medical points including treatments, medications, side effects, risks, symptoms, dosages, and clinical findings. Keep the summary concise but comprehensive:

{text[:4000]}  # Limit text length for API

Summary:"""

        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logger.error(f"Error summarizing text: {e}")
        return text[:1000]  # Fallback to truncated original text

if __name__ == '__main__':
    # Initialize everything
    initialize_ai()
    load_documents()
    load_index()
    
    # If no index exists, build one
    if index is None and documents:
        build_index()
    
    app.run(debug=True, host='127.0.0.1', port=5000)