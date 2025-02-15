Filename: src/locallama.py
Copyright: (c) 2025 Gregory L. Magnusson MIT License

Summary:
This file implements the OllamaHandler class, which serves as an interface to the Ollama API for local LLM interactions. Key features include:

Core Components:
OllamaResponse dataclass for structured API responses
OllamaHandler class for managing Ollama API interactions
Default Deepseek model configurations
Main Functionalities:
Model management (listing, selection, pulling)
Response generation (streaming and non-streaming)
Temperature control
Embedding generation
Error handling and logging
Auto-selection of default models
API Integration:
RESTful API calls to Ollama service
Support for both synchronous and streaming responses
Embedding API integration
Model verification and pulling

locallama.py provides a comprehensive wrapper around the Ollama API, with robust error handling and configuration options. It's designed to work primarily with Deepseek models but can handle other Ollama-compatible models as well. The implementation includes modern Python features like dataclasses, type hints, and generators for streaming responses.

Filename: src/logger.py
Copyright: (c) 2025 Gregory L. Magnusson MIT license

Summary:
This file implements a sophisticated logging system for the RAGE project with the following key features:

Core Components:
StructuredFormatter: Custom formatter for structured log data
ContextLogger: Thread-safe logger with context binding
Setup functionality for configuring logging instances
Key Features:
Rotating file handlers for both general and error logs
Thread-safe context management
JSON-structured logging support
Size-based log rotation (default 10MB)
Separate error logging stream
Implementation Details:
Multiple output streams (console and file)
Context binding capability for adding metadata to logs
Filtered structured data with allowed keys
UTF-8 encoding support
Thread-local storage for context isolation
The logger is designed to provide comprehensive logging capabilities with structured data support, making it suitable for debugging and monitoring in a production environment. It includes features like log rotation, separate error logging, and thread-safe context management, making it a robust solution for the RAGE project's logging needs.

Filename: memory.py
Copyright: (c) 2025 Gregory L. Magnusson MIT license

Summary:
implements the memory management system for RAGE, providing both short-term and long-term memory capabilities. Key features include:

Core Components:
ContextType Enum: Defines different types of context (conversation, knowledge, code, web)
ContextEntry Dataclass: Structured format for memory entries
MemoryManager Class: Singleton class managing memory operations
Memory Architecture:
Short-term Memory (STM): Implemented as a deque with fixed capacity (20 entries)
Long-term Memory (LTM): Persistent storage using JSON files
Session-based conversation storage
Thread-safe operations
Key Functionalities:
Context storage and retrieval
Relevance scoring for context retrieval
Atomic file operations for data persistence
Directory structure management
Conversation tracking by sessions
Memory serialization and deserialization

memory.py provides a robust memory system with thread safety, error handling, and efficient context retrieval capabilities. It supports different types of context and includes utilities for storing both conversations and knowledge entries, making it a crucial component for RAGE context-aware operations.

Filename: src/openmind.py
Copyright: (c) 2025 Gregory L. Magnusson MIT license

Summary:
This file implements the OpenMind class, which manages the prompt system for RAGE. It's designed as a singleton pattern to handle central configuration and resource management. Key features include:

Core Functionality:
Singleton pattern implementation
Prompt file loading and management
UTF-8 encoding support
Error handling for file operations
Main Components:
System prompt loading from 'system_prompt.txt'
User prompt loading from 'prompt.txt'
Path resolution for prompt files
Logging integration
Key Methods:
_load_prompt: Handles prompt file loading with error handling
get_system_prompt: Returns the system prompt
get_user_prompt: Returns the user prompt template
The implementation is straightforward but crucial for RAGE's operation, as it manages the prompts that guide the AI's behavior and responses. It includes proper error handling and logging, ensuring robust operation in a production environment.


