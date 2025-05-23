import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load environment variables from .env file
load_dotenv()

class Config:
    """Configuration settings for the RAG LLM Backend"""
    
    # Server Configuration
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    
    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    
    # Database Configuration
    CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
    
    # Document Processing Configuration
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 1000))
    CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))
    MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
    
    # Upload Configuration
    UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "uploads"))
    ALLOWED_EXTENSIONS = {'.pdf', '.txt', '.docx', '.doc', '.md'}
    
    # Logging Configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    # RAG Configuration
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", 4))  # Number of documents to retrieve
    
    @classmethod
    def setup_logging(cls):
        """Setup logging configuration"""
        logging.basicConfig(
            level=getattr(logging, cls.LOG_LEVEL),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('app.log')
            ]
        )
    
    @classmethod
    def validate_config(cls):
        """Validate configuration and API keys"""
        errors = []
        
        # Check if at least one API key is provided
        api_keys = [cls.OPENAI_API_KEY, cls.ANTHROPIC_API_KEY, cls.GOOGLE_API_KEY]
        if not any(api_keys):
            errors.append("At least one LLM API key must be provided")
        
        # Check if OpenAI key is available (required for embeddings)
        if not cls.OPENAI_API_KEY:
            errors.append("OpenAI API key is required for embeddings")
        
        # Create necessary directories
        cls.UPLOAD_DIR.mkdir(exist_ok=True)
        Path(cls.CHROMA_DB_PATH).parent.mkdir(exist_ok=True)
        
        if errors:
            raise ValueError(f"Configuration errors: {'; '.join(errors)}")
        
        return True

# Initialize configuration
Config.setup_logging()
Config.validate_config()