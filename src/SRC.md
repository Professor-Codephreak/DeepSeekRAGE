Filename: src/locallama.py
Copyright: (c) 2025 Gregory L. Magnusson MIT License

Summary:
This file implements the OllamaHandler class, which serves as an interface to the Ollama API for local LLM interactions. Key features include:<br />

Core Components:<br />
OllamaResponse dataclass for structured API responses<br />
OllamaHandler class for managing Ollama API interactions<br />
Default Deepseek model configurations<br />
Main Functionalities:<br />
Model management (listing, selection, pulling)<br />
Response generation (streaming and non-streaming)<br />
Temperature control<br />
Embedding generation<br />
Error handling and logging<br />
Auto-selection of default models<br />
API Integration:<br />
RESTful API calls to Ollama service<br />
Support for both synchronous and streaming responses<br />
Embedding API integration<br />
Model verification and pulling

locallama.py provides a comprehensive wrapper around the Ollama API, with robust error handling and configuration options. It's designed to work primarily with Deepseek models but can handle other Ollama-compatible models as well. The implementation includes modern Python features like dataclasses, type hints, and generators for streaming responses.<br />

Filename: src/logger.py<br />
Copyright: (c) 2025 Gregory L. Magnusson MIT license<br />

Summary:
This file implements a sophisticated logging system for the RAGE project with the following key features:<br />

Core Components:
StructuredFormatter: Custom formatter for structured log data<br />
ContextLogger: Thread-safe logger with context binding<br />
Setup functionality for configuring logging instances<br />
Key Features:<br />
Rotating file handlers for both general and error logs<br />
Thread-safe context management<br />
JSON-structured logging support<br />
Size-based log rotation (default 10MB)<br />
Separate error logging stream<br />
Implementation Details:<br />
Multiple output streams (console and file)<br />
Context binding capability for adding metadata to logs<br />
Filtered structured data with allowed keys<br />
UTF-8 encoding support<br />
Thread-local storage for context isolation<br />
The logger is designed to provide comprehensive logging capabilities with structured data support, making it suitable for debugging and monitoring in a production environment. It includes features like log rotation, separate error logging, and thread-safe context management, making it a robust solution for the RAGE project's logging needs.<br />

Filename: memory.py<br />
Copyright: (c) 2025 Gregory L. Magnusson MIT license<br />

Summary:
implements the memory management system for RAGE, providing both short-term and long-term memory capabilities. Key features include:<br />

ContextType Enum: Defines different types of context (conversation, knowledge, code, web)<br />
ContextEntry Dataclass: Structured format for memory entries<br />
MemoryManager Class: Singleton class managing memory operations<br />
Memory Architecture:<br />
Short-term Memory (STM): Implemented as a deque with fixed capacity (20 entries)<br />
Long-term Memory (LTM): Persistent storage using JSON files<br />
Session-based conversation storage<br />
Thread-safe operations<br />
Key Functionalities:<br />
Context storage and retrieval<br />
Relevance scoring for context retrieval<br />
Atomic file operations for data persistence<br />
Directory structure management<br />
Conversation tracking by sessions<br />
Memory serialization and deserialization<br />

memory.py provides a robust memory system with thread safety, error handling, and efficient context retrieval capabilities. It supports different types of context and includes utilities for storing both conversations and knowledge entries, making it a crucial component for RAGE context-aware operations.<br />

Filename: src/openmind.py<br />
Copyright: (c) 2025 Gregory L. Magnusson MIT license<br />

Summary:
This file implements the OpenMind class, which manages the prompt system for RAGE. It's designed as a singleton pattern to handle central configuration and resource management. Key features include:<br />

Singleton pattern implementation<br />
Prompt file loading and management<br />
UTF-8 encoding support<br />
Error handling for file operations<br />
Main Components:<br />
System prompt loading from 'system_prompt.txt'<br />
User prompt loading from 'prompt.txt'<br />
Path resolution for prompt files<br />
Logging integration<br />
Key Methods:<br />
_load_prompt: Handles prompt file loading with error handling<br />
get_system_prompt: Returns the system prompt<br />
get_user_prompt: Returns the user prompt template<br />
The implementation is straightforward but crucial for RAGE's operation, as it manages the prompts that guide the AI's behavior and responses. It includes proper error handling and logging, ensuring robust operation in a production environment.<br />


