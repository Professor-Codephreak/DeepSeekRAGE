# src/memory.py (c) 2025 Gregory L. Magnusson MIT license

import os
import json
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import hashlib
from pathlib import Path
import logging
from tqdm import tqdm
import requests
import streamlit as st

logger = logging.getLogger('rage.memory')

@dataclass
class DialogEntry:
    """Structure for dialogue entries"""
    query: str
    response: str
    timestamp: str = None
    context: Dict = None
    provider: str = None
    model: str = None
    sources: List[Dict] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.context is None:
            self.context = {}
        if self.sources is None:
            self.sources = []

@dataclass
class MemoryEntry:
    """Structure for memory entries"""
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    entry_id: Optional[str] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.entry_id is None:
            self.entry_id = hashlib.md5(
                f"{self.timestamp}{self.content}".encode()
            ).hexdigest()

class MemoryManager:
    """Memory management system for RAGE"""
    
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MemoryManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.logger = logging.getLogger('rage.memory')
        
        # Define memory structure
        self.base_dir = Path('./memory')
        self.memory_structure = {
            'conversations': self.base_dir / 'conversations',
            'knowledge': self.base_dir / 'knowledge',
            'cache': self.base_dir / 'cache',
            'index': self.base_dir / 'index'
        }
        
        # Initialize system
        self._initialize_memory_system()
        
        # Create session file
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._create_session_file()
        
        # Initialize documents list
        self.documents = []
        self._load_existing_documents()

    def _initialize_memory_system(self):
        """Initialize memory system and create directories"""
        try:
            # Create all directories
            for directory in self.memory_structure.values():
                directory.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Memory system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory system: {e}")
            raise

    def _create_session_file(self):
        """Create session tracking file"""
        session_data = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "entries": []
        }
        
        session_file = self.memory_structure['conversations'] / f"session_{self.session_id}.json"
        self._write_json(session_file, session_data)

    def _write_json(self, filepath: Path, data: Dict) -> bool:
        """Write data to JSON file with error handling"""
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"Failed to write JSON file {filepath}: {e}")
            return False

    def _read_json(self, filepath: Path) -> Optional[Dict]:
        """Read JSON file with error handling"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to read JSON file {filepath}: {e}")
            return None

    def _load_existing_documents(self):
        """Load existing documents if available"""
        try:
            docs_path = self.memory_structure['knowledge'] / "documents.json"
            if docs_path.exists():
                docs_data = self._read_json(docs_path)
                if docs_data:
                    self.documents = [MemoryEntry(**doc) for doc in docs_data]
                self.logger.info(f"Loaded {len(self.documents)} documents from existing knowledge base")
        except Exception as e:
            self.logger.error(f"Error loading existing documents: {e}")
            self.documents = []

    def get_relevant_context(self, query: str, k: int = 3) -> str:
        """Get relevant context for query (simplified without embeddings)"""
        try:
            if not self.documents:
                return ""
            
            # Simple keyword-based search (replace with a better search mechanism if needed)
            results = [
                doc for doc in self.documents
                if query.lower() in doc.content.lower()
            ][:k]
            
            # Combine content from relevant documents
            context = "\n\n".join([
                f"Source {i+1}:\n{doc.content}"
                for i, doc in enumerate(results)
            ])
            
            self.logger.info(f"Retrieved context from {len(results)} documents")
            return context
            
        except Exception as e:
            self.logger.error(f"Error getting relevant context: {e}")
            return ""

    def store_conversation(self, entry: DialogEntry) -> bool:
        """Store conversation entry"""
        try:
            # Generate unique ID
            entry_id = hashlib.md5(
                f"{entry.timestamp}{entry.query}".encode()
            ).hexdigest()
            
            # Prepare entry data
            entry_data = asdict(entry)
            entry_data.update({
                'entry_id': entry_id,
                'session_id': self.session_id
            })
            
            # Save conversation entry
            conv_file = self.memory_structure['conversations'] / f"conv_{entry_id}.json"
            if self._write_json(conv_file, entry_data):
                # Update session file
                session_file = self.memory_structure['conversations'] / f"session_{self.session_id}.json"
                session_data = self._read_json(session_file)
                if session_data:
                    session_data['entries'].append(entry_id)
                    self._write_json(session_file, session_data)
                
                self.logger.info(f"Stored conversation entry: {entry_id}")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to store conversation: {e}")
            return False

    def store_memory(self, entry: MemoryEntry) -> bool:
        """Store memory entry (without embeddings)"""
        try:
            # Save memory entry
            memory_file = self.memory_structure['knowledge'] / f"memory_{entry.entry_id}.json"
            memory_data = asdict(entry)
            
            # Add to documents list
            self.documents.append(entry)
            
            # Save documents data
            docs_path = self.memory_structure['knowledge'] / "documents.json"
            self._write_json(docs_path, [asdict(doc) for doc in self.documents])
            
            return self._write_json(memory_file, memory_data)
            
        except Exception as e:
            self.logger.error(f"Failed to store memory: {e}")
            return False

    def get_conversation_history(self, 
                               session_id: Optional[str] = None,
                               limit: int = 100) -> List[Dict]:
        """Get conversation history"""
        try:
            session_id = session_id or self.session_id
            
            # Get session file
            session_file = self.memory_structure['conversations'] / f"session_{session_id}.json"
            session_data = self._read_json(session_file)
            
            if session_data and 'entries' in session_data:
                conversations = []
                for entry_id in session_data['entries'][-limit:]:
                    conv_file = self.memory_structure['conversations'] / f"conv_{entry_id}.json"
                    if conv_data := self._read_json(conv_file):
                        conversations.append(conv_data)
                
                return conversations
            
            return []
            
        except Exception as e:
            self.logger.error(f"Error retrieving conversation history: {e}")
            return []

    def clear_cache(self) -> bool:
        """Clear cache directory"""
        try:
            cache_dir = self.memory_structure['cache']
            for file in cache_dir.glob("*"):
                file.unlink()
            self.logger.info("Cache cleared successfully")
            return True
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False

# Global instance
memory_manager = MemoryManager()

# Convenience functions
def store_conversation(entry: DialogEntry) -> bool:
    """Store conversation entry"""
    return memory_manager.store_conversation(entry)

def store_memory(entry: MemoryEntry) -> bool:
    """Store memory entry"""
    return memory_manager.store_memory(entry)

def get_conversation_history(
    session_id: Optional[str] = None,
    limit: int = 100
) -> List[Dict]:
    """Get conversation history"""
    return memory_manager.get_conversation_history(session_id, limit)

def clear_cache() -> bool:
    """Clear cache"""
    return memory_manager.clear_cache()
