# RAG LLM Backend

A FastAPI-based backend for RAG (Retrieval-Augmented Generation) with support for multiple LLMs including OpenAI GPT, Anthropic Claude, and Google Gemini.

## Features

- 📄 **Multi-format document support**: PDF, TXT, DOCX, MD files
- 🤖 **Multiple LLM support**: OpenAI, Claude, Gemini with random selection
- 🔍 **Advanced RAG**: Vector similarity search with ChromaDB
- 📊 **Document processing**: Automatic text chunking and embedding
- 🚀 **Fast API**: RESTful API with automatic documentation
- 🔒 **Secure**: Environment-based API key management

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <your-repo>
cd rag-llm-backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:

```env
# Required: OpenAI API Key (for embeddings)
OPENAI_API_KEY=sk-your-openai-key-here

# Optional: Additional LLMs (add any or all)
ANTHROPIC_API_KEY=your-anthropic-key-here
GOOGLE_API_KEY=your-google-key-here
```

### 3. Run the Backend

```bash
# Using the startup script (recommended)
python run.py

# Or directly with uvicorn
uvicorn main:app --reload
```

The backend will start on `http://localhost:8000`

## API Endpoints

### Document Management

- **POST /upload** - Upload documents for processing
- **GET /documents** - List all processed documents
- **DELETE /documents** - Clear all documents

### Question Answering

- **POST /ask** - Ask questions and get RAG-based answers

### System Status

- **GET /health** - Check system health and available LLMs
- **GET /status** - Get current system status

## API Usage Examples

### Upload Documents

```bash
curl -X POST "http://localhost:8000/upload" \
  -F "files=@document1.pdf" \
  -F "files=@document2.txt"
```

### Ask Questions

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the main topic of the documents?"}'
```

### Check Status

```bash
curl "http://localhost:8000/status"
```

## Frontend Integration

Update your frontend JavaScript to use the backend API:

```javascript
// Upload documents
async function uploadDocuments(files) {
    const formData = new FormData();
    files.forEach(file => formData.append('files', file));
    
    const response = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData
    });
    
    return response.json();
}

// Ask questions
async function askQuestion(question) {
    const response = await fetch('http://localhost:8000/ask', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ question })
    });
    
    return response.json();
}
```

## Configuration

All configuration is handled through environment variables in the `.env` file:

### Required Settings

- `OPENAI_API_KEY`: Required for embeddings and optionally for LLM responses

### Optional LLM Settings

- `ANTHROPIC_API_KEY`: Enable Claude LLM
- `GOOGLE_API_KEY`: Enable Gemini LLM

### Optional Server Settings

- `HOST`: Server host (default: 0.0.0.0)
- `PORT`: Server port (default: 8000)
- `DEBUG`: Enable debug mode (default: False)

### Optional Processing Settings

- `CHUNK_SIZE`: Text chunk size (default: 1000)
- `CHUNK_OVERLAP`: Chunk overlap (default: 200)
- `RETRIEVAL_K`: Number of documents to retrieve (default: 4)

## File Structure

```
rag-llm-backend/
├── main.py              # FastAPI application
├── rag_service.py       # RAG implementation
├── llm_manager.py       # LLM management
├── document_processor.py # Document processing
├── config.py            # Configuration management
├── run.py              # Startup script
├── requirements.txt     # Python dependencies
├── .env.example        # Environment template
├── .env                # Your API keys (create this)
└── README.md           # This file
```

## Supported File Types

- **PDF**: `.pdf` files
- **Text**: `.txt` files
- **Word**: `.docx`, `.doc` files
- **Markdown**: `.md` files

## How It Works

1. **Document Upload**: Files are uploaded and processed using appropriate loaders
2. **Text Splitting**: Documents are split into chunks for better retrieval
3. **Embeddings**: Text chunks are converted to vector embeddings using OpenAI
4. **Vector Storage**: Embeddings are stored in ChromaDB for fast similarity search
5. **Question Processing**: Questions are embedded and similar documents are retrieved
6. **LLM Selection**: A random LLM is selected from available options
7. **Answer Generation**: The selected LLM generates an answer using retrieved context

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Troubleshooting

### Common Issues

1. **No LLM API keys**: Make sure you have at least one API key in your `.env` file
2. **OpenAI key missing**: OpenAI is required for embeddings
3. **Port already in use**: Change the PORT in your `.env` file
4. **File upload fails**: Check file format and size limits

### Logging

Logs are written to both console and `app.log` file. Set `LOG_LEVEL=DEBUG` for detailed logs.

### Getting API Keys

- **OpenAI**: https://platform.openai.com/api-keys
- **Anthropic**: https://console.anthropic.com/
- **Google**: https://makersuite.google.com/app/apikey

## Development

### Running in Development Mode

```bash
# With auto-reload
python run.py

# Or with uvicorn directly
uvicorn main:app --reload --log-level debug
```

### Testing

```bash
# Test health endpoint
curl http://localhost:8000/health

# Test with sample document
curl -X POST "http://localhost:8000/upload" \
  -F "files=@sample.txt"

# Test question
curl -X POST "http://localhost:8000/ask" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'
```

## Production Deployment

For production deployment, consider:

1. **Environment Variables**: Use proper environment variable management
2. **Database**: Consider using persistent vector databases
3. **Security**: Add authentication and rate limiting
4. **Scaling**: Use proper ASGI server like Gunicorn with Uvicorn workers
5. **Monitoring**: Add logging and monitoring solutions

## License

This project is licensed under the MIT License.