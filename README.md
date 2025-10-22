# Medical Evidence Retrieval System

A modern, AI-powered medical evidence retrieval system that provides evidence-based answers with semantic search capabilities. Built with Flask, Google Gemini AI, and FAISS vector search for intelligent medical document processing and querying.

## ✨ Key Features

### 🔍 Intelligent Search
- **Semantic Search**: Uses FAISS and Gemini embeddings for cosine similarity search across medical documents
- **Evidence-Based Answers**: Returns concise, patient-friendly answers (2-3 sentences) with proper citations
- **Source Attribution**: Displays 2-3 relevant sources with confidence scores and snippets

### 📚 Advanced Document Processing
- **Multi-Format Support**: Upload PDF, CSV, and TXT files with drag-and-drop functionality
- **AI-Powered Summarization**: Automatically extracts key medical points (treatments, medications, side effects, risks, symptoms, dosages, clinical findings)
- **Text Ingestion**: Paste medical abstracts and content directly with intelligent summarization
- **Batch Processing**: Upload multiple files simultaneously

### 🎨 Modern User Interface
- **Responsive Design**: Beautiful, mobile-friendly interface built with Tailwind CSS
- **Dark/Light Mode**: Toggle between themes with persistent preferences
- **Interactive Chat**: Real-time conversation interface with typing indicators
- **Collapsible Panels**: Clean, organized layout with expandable sections
- **Visual Feedback**: Loading states, success/error messages, and progress indicators

### 🧠 AI Integration
- **Google Gemini API**: Advanced language model for content summarization and answer generation
- **Vector Embeddings**: Efficient semantic search using Google's embedding models
- **Medical Focus**: Specialized prompts for extracting medical information

## 🛠️ Tech Stack

### Backend
- **Python 3.11+**: Core runtime environment
- **Flask**: Web framework with CORS support
- **FAISS**: Vector similarity search and indexing
- **Google Generative AI**: Gemini API for embeddings and text generation
- **PyPDF2**: PDF text extraction and processing
- **NumPy**: Numerical operations for vector processing

### Frontend
- **HTML5**: Semantic markup structure
- **Tailwind CSS**: Utility-first CSS framework
- **Vanilla JavaScript**: Interactive functionality and API communication
- **Font Awesome**: Icon library for UI elements

### Data Storage
- **JSON**: File-based document storage (diabetes_abstracts.json)
- **FAISS Index**: Binary vector index for fast similarity search
- **File System**: Local storage for uploaded documents

## 📂 Project Structure

```
├── app.py                    # Flask backend (API + FAISS search)
├── index.html                # Modern frontend interface
├── diabetes_abstracts.json   # Preloaded medical dataset
├── faiss_index.bin          # Vector search index
├── requirements.txt          # Python dependencies
├── uploads/                  # Uploaded document storage
└── README.md                 # Documentation
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Google Gemini API key
- Git (for cloning)

### Step 1: Clone Repository
```bash
git clone https://github.com/Hjmontalban/Medical-Diabetes-Agent.git
cd Medical-Diabetes-Agent
```

### Step 2: Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure API Key
The application uses a hardcoded API key for convenience. Replace it with your own key in `app.py`:

```python
# Replace the hardcoded key in app.py
API_KEY = "your-gemini-api-key-here"
```

### Step 5: Run Application
```bash
python app.py
```

The application will be available at: **http://127.0.0.1:5000/**

## 📖 Usage Guide

### Getting Started
1. Open **http://127.0.0.1:5000/** in your browser
2. The interface loads with a pre-populated medical knowledge base
3. Start asking medical questions in the chat interface

### Asking Questions
Ask evidence-based medical questions such as:
- "What are the latest treatments for type 2 diabetes?"
- "What are common side effects of GLP-1 receptor agonists?"
- "What risks are associated with gestational diabetes?"
- "What are the benefits of metformin therapy?"

### Adding Medical Sources

#### Method 1: File Upload
1. Click **"Add Medical Sources"** to expand the panel
2. Select the **"Upload Files"** tab
3. **Drag & drop** files or click **"Choose Files"**
4. Supported formats: PDF, CSV, TXT
5. Add optional title and citation
6. Click **"Upload and Index Files"**

#### Method 2: Text Input
1. Click **"Add Medical Sources"** to expand the panel
2. Select the **"Paste Text"** tab
3. Enter a **title** (required)
4. **Paste medical content** (abstracts, research text, etc.)
5. Add optional **citation**
6. Click **"Summarize and Index Content"**

### Understanding Results
- **AI Answer**: 2-3 sentence patient-friendly response
- **Sources**: 2-3 relevant documents with:
  - Document title
  - Citation information
  - Content snippet
  - Confidence score

## 📊 Sample Data Structure

### Document Format
Each document in the knowledge base follows this structure:

```json
{
  "id": "doc1",
  "title": "Metformin and Glycemic Control",
  "text": "AI-summarized medical content focusing on key points...",
  "citation": "Smith et al., 2010"
}
```

## 🔧 API Endpoints

### POST /api/search
Search for medical information
```json
{
  "query": "What are the side effects of metformin?"
}
```

### POST /api/upload
Upload and process files
- Multipart form data with files
- Optional title and citation fields

### POST /api/add_text
Add text content directly
```json
{
  "title": "Document Title",
  "text": "Medical content...",
  "citation": "Optional citation"
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support, please open an issue on GitHub or contact the development team.

## 🔮 Future Enhancements

- Multi-language support
- Advanced filtering options
- User authentication
- Document versioning
- Export functionality
- Integration with medical databases
