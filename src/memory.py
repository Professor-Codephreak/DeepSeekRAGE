# src/memory.py
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Deque
from dataclasses import dataclass, asdict, field
from collections import deque
import hashlib
import logging
from enum import Enum
import threading

logger = logging.getLogger('rage.memory')

# Constants
MAX_STM_ENTRIES = 20  # Short-term memory capacity

class ContextType(Enum):
    CONVERSATION = "conversation"
    KNOWLEDGE = "knowledge"
    CODE = "code"
    WEB = "web"

@dataclass
class ContextEntry:
    """Unified context entry structure"""
    content: str
    context_type: ContextType
    source: str = "system"
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    weight: float = 1.0  # Relevance weight (0-1)

    def __post_init__(self):
        self.id = hashlib.md5(
            f"{self.timestamp}{self.content}".encode()
        ).hexdigest()

    def to_dict(self):
        """Convert the ContextEntry to a dictionary with serializable values."""
        data = asdict(self)
        data['context_type'] = self.context_type.value  # Convert enum to string
        data['timestamp'] = self.timestamp.isoformat()  # Convert datetime to string
        return data

class MemoryManager:
    """Enhanced memory manager with short/long-term memory"""
    
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._init_done = False
        return cls._instance

    def __init__(self):
        if self._init_done:
            return
            
        self._init_done = True
        self.base_dir = Path('./memory')
        self._setup_directories()
        
        # Short-term memory (recent context)
        self.stm: Deque[ContextEntry] = deque(maxlen=MAX_STM_ENTRIES)
        
        # Long-term memory storage
        self.ltm_path = self.base_dir / 'long_term_memory.json'
        self.ltm_lock = threading.Lock()
        self.ltm = []  # Initialize ltm as an empty list
        self._load_ltm()  # Load existing long-term memory

    def _setup_directories(self):
        """Create required memory directories"""
        (self.base_dir / 'sessions').mkdir(exist_ok=True)
        (self.base_dir / 'knowledge').mkdir(exist_ok=True)

    def _load_ltm(self):
        """Load long-term memory with error handling"""
        try:
            if self.ltm_path.exists():
                with open(self.ltm_path, 'r') as f:
                    data = json.load(f)
                    self.ltm = [ContextEntry(**entry) for entry in data]
                logger.info(f"Loaded {len(self.ltm)} long-term memory entries")
        except Exception as e:
            logger.error(f"Failed to load LTM: {e}")
            self.ltm = []

    def _save_ltm(self):
        """Save long-term memory atomically"""
        with self.ltm_lock:
            temp_path = self.ltm_path.with_suffix('.tmp')
            try:
                with open(temp_path, 'w') as f:
                    json.dump([entry.to_dict() for entry in self.ltm], f)
                temp_path.replace(self.ltm_path)
                logger.info(f"Saved {len(self.ltm)} long-term memory entries")
            except Exception as e:
                logger.error(f"LTM save failed: {e}")

    def add_context(self, entry: ContextEntry):
        """Add context to memory with proper storage"""
        # Always keep in short-term memory
        self.stm.append(entry)
        
        # Persist to appropriate long-term storage
        if entry.context_type == ContextType.CONVERSATION:
            self._store_conversation(entry)
        else:
            with self.ltm_lock:
                self.ltm.append(entry)
                self._save_ltm()

    def _store_conversation(self, entry: ContextEntry):
        """Store conversation context with session tracking"""
        session_file = self.base_dir / 'sessions' / f"session_{datetime.now():%Y%m%d}.json"
        
        try:
            with open(session_file, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + '\n')
            logger.info(f"Stored conversation entry: {entry.id}")
        except Exception as e:
            logger.error(f"Conversation storage failed: {e}")

    def get_relevant_context(self, query: str, max_results: int = 7) -> str:
        """Retrieve relevant context from all memory sources"""
        # Combine STM and LTM with basic relevance scoring
        all_context = list(self.stm) + self.ltm
        
        # Simple keyword-based relevance scoring
        scored_context = []
        query_lower = query.lower()
        
        for entry in all_context:
            score = self._calculate_relevance(entry.content, query_lower)
            scored_context.append((score, entry))
        
        # Sort by score and return top results
        scored_context.sort(reverse=True, key=lambda x: x[0])
        return self._format_context(scored_context[:max_results])

    def _calculate_relevance(self, content: str, query: str) -> float:
        """Basic relevance scoring algorithm"""
        content_lower = content.lower()
        
        # Simple term frequency scoring
        term_count = sum(1 for word in query.split() if word in content_lower)
        return term_count / len(query.split()) if query else 0

    def _format_context(self, context: List) -> str:
        """Format context for LLM consumption"""
        return '\n\n'.join(
            f"[{entry.context_type.value.upper()}] {entry.content}"
            for score, entry in context
            if score > 0
        )

    def clear_stm(self):
        """Clear short-term memory"""
        self.stm.clear()
        logger.info("Cleared short-term memory")

# Global instance with thread safety
memory_manager = MemoryManager()

# Utility functions
def store_conversation(entry: ContextEntry) -> bool:
    """Store conversation entry"""
    return memory_manager.add_context(entry)

def store_knowledge(text: str, metadata: Dict = None) -> ContextEntry:
    """Store knowledge context"""
    entry = ContextEntry(
        content=text,
        context_type=ContextType.KNOWLEDGE,
        metadata=metadata or {},
        source="system"
    )
    memory_manager.add_context(entry)
    return entry
