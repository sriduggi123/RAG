import os
import random
from typing import Dict, List, Optional
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.language_models import BaseChatModel
import logging

logger = logging.getLogger(__name__)

class LLMManager:
    """Manages multiple LLM providers and randomly selects one for each request"""
    
    def __init__(self):
        self.llms: Dict[str, BaseChatModel] = {}
        self.available_llms: List[str] = []
        self._initialize_llms()
    
    def _initialize_llms(self):
        """Initialize all available LLMs based on API keys in environment"""
        
        # OpenAI GPT
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if openai_api_key:
            try:
                self.llms["openai"] = ChatOpenAI(
                    api_key=openai_api_key,
                    model="gpt-3.5-turbo",
                    temperature=0.7,
                    max_tokens=1000
                )
                self.available_llms.append("openai")
                logger.info("OpenAI LLM initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI LLM: {e}")
        
        # Anthropic Claude
        anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        if anthropic_api_key:
            try:
                self.llms["claude"] = ChatAnthropic(
                    api_key=anthropic_api_key,
                    model="claude-3-haiku-20240307",
                    temperature=0.7,
                    max_tokens=1000
                )
                self.available_llms.append("claude")
                logger.info("Claude LLM initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Claude LLM: {e}")
        
        # Google Gemini
        google_api_key = os.getenv("GOOGLE_API_KEY")
        if google_api_key:
            try:
                self.llms["gemini"] = ChatGoogleGenerativeAI(
                    api_key=google_api_key,
                    model="gemini-pro",
                    temperature=0.7,
                    max_output_tokens=1000
                )
                self.available_llms.append("gemini")
                logger.info("Gemini LLM initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini LLM: {e}")
        
        if not self.available_llms:
            raise ValueError("No LLM API keys found. Please check your .env file.")
        
        logger.info(f"Initialized {len(self.available_llms)} LLMs: {', '.join(self.available_llms)}")
    
    def get_random_llm(self) -> tuple[BaseChatModel, str]:
        """Get a randomly selected LLM"""
        if not self.available_llms:
            raise ValueError("No LLMs available")
        
        selected_llm = random.choice(self.available_llms)
        return self.llms[selected_llm], selected_llm
    
    def get_specific_llm(self, llm_name: str) -> Optional[BaseChatModel]:
        """Get a specific LLM by name"""
        return self.llms.get(llm_name)
    
    def get_available_llms(self) -> List[str]:
        """Get list of available LLM names"""
        return self.available_llms.copy()
    
    def is_available(self, llm_name: str) -> bool:
        """Check if a specific LLM is available"""
        return llm_name in self.available_llms