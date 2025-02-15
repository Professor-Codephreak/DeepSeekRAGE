# src/openmind.py (c) 2025 Gregory L. Magnusson MIT license
# OpenMind class to loads prompt.txt and system_prompt.txt from prompts
import os
from pathlib import Path
import logging

logger = logging.getLogger('rage.openmind')

class OpenMind:
    """central configuration and resource management for RAGE"""
    
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.base_path = Path(__file__).parent.parent.resolve()
        self.prompts_path = self.base_path / 'prompts'
        
        # Load prompts from prompts/prompts.txt and prompts/system_prompt.txt
        self.system_prompt = self._load_prompt('system_prompt.txt')
        self.user_prompt = self._load_prompt('prompt.txt')
        
        logger.info("OpenMind initialized successfully")

    def _load_prompt(self, filename: str) -> str:
        """Load a prompt file with error handling."""
        try:
            prompt_file = self.prompts_path / filename
            if not prompt_file.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
            
            with open(prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Failed to load prompt {filename}: {e}")
            raise

    def get_system_prompt(self) -> str:
        """Get the system prompt."""
        return self.system_prompt

    def get_user_prompt(self) -> str:
        """Get the user prompt template."""
        return self.user_prompt
