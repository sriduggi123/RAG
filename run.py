#!/usr/bin/env python3
"""
Startup script for the RAG LLM Backend
"""

import sys
import os
import logging
from pathlib import Path

# Add current directory to path
sys.path.append(str(Path(__file__).parent))

def main():
    """Main startup function"""
    print("🚀 Starting RAG LLM Backend...")
    
    # Check requirements
    #if not check_requirements():
    #    sys.exit(1)
    
    # Check environment
    if not check_env_file():
        response = input("Do you want to continue anyway? (y/N): ")
        if response.lower() != 'y':
            sys.exit(1)

    try:
        # Import and start the application
        from config import Config
        import uvicorn
        from main import app
        
        print(f"✅ Configuration validated")
        print(f"🌐 Starting server on {Config.HOST}:{Config.PORT}")
        print(f"📚 Vector database: {Config.CHROMA_DB_PATH}")
        print(f"📁 Upload directory: {Config.UPLOAD_DIR}")
        
        # Start the server
        uvicorn.run(
            app,
            host=Config.HOST,
            port=Config.PORT,
            reload=Config.DEBUG,
            log_level=Config.LOG_LEVEL.lower()
        )
        
    except Exception as e:
        logging.error(f"Failed to start server: {e}")
        sys.exit(1)

def check_requirements():
    """Check if all required packages are installed"""
    required_packages = [
        'fastapi', 'uvicorn', 'langchain', 'chromadb', 'pypdf', 'docx2txt'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path('.env')
    
    if not env_file.exists():
        print("Warning: .env file not found!")
        print("Please create a .env file based on .env.example")
        print("At minimum, you need to provide API keys for the LLMs")
        return False
    
    # Check for at least one API key
    from dotenv import load_dotenv
    load_dotenv()
    
    api_keys = [
        os.getenv("OPENAI_API_KEY"),
        os.getenv("ANTHROPIC_API_KEY"), 
        os.getenv("GOOGLE_API_KEY")
    ]
    
    if not any(api_keys):
        print("Warning: No LLM API keys found in .env file!")
        print("Please add at least one API key to your .env file")
        return False
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OpenAI API key not found!")
        print("OpenAI API key is required for embeddings")
        return False
    
    return True


if __name__ == "__main__":
    main() 
    

