# src/logger.py (c) 2025 Gregory L. Magnusson
import logging
import sys
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional, Set
from logging.handlers import RotatingFileHandler
import threading

# Thread-local storage for context
_thread_local = threading.local()

class StructuredFormatter(logging.Formatter):
    """Clean formatter with structured data support"""
    
    def __init__(self):
        super().__init__(
            fmt='%(asctime)s | %(levelname)8s | %(name)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        self.allowed_keys = {'session_id', 'model', 'provider', 'error', 'duration'}

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with structured data"""
        base_message = super().format(record)
        structured = getattr(record, 'structured_data', {})
        
        # Filter and format structured data
        filtered = {
            k: v for k, v in structured.items() 
            if k in self.allowed_keys and self._is_serializable(v)
        }
        
        if filtered:
            return f"{base_message} || {json.dumps(filtered, default=str)}"
        return base_message

    def _is_serializable(self, value: Any) -> bool:
        """Check if value can be JSON serialized"""
        try:
            json.dumps(value)
            return True
        except (TypeError, ValueError):
            return False

def setup_logger(
    name: str,
    log_dir: Path = Path("logs"),
    max_size: int = 10 * 1024 * 1024  # 10MB
) -> logging.Logger:
    """Configure a logger with isolated error handling"""
    
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates
    if logger.hasHandlers():
        logger.handlers.clear()

    # Console handler for general output
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(StructuredFormatter())
    logger.addHandler(console)

    # Create log directory if needed
    log_dir.mkdir(parents=True, exist_ok=True)

    # Regular file handler (all messages)
    main_handler = RotatingFileHandler(
        filename=log_dir / f"{name}.log",
        maxBytes=max_size,
        backupCount=3,
        encoding='utf-8'
    )
    main_handler.setFormatter(StructuredFormatter())
    logger.addHandler(main_handler)

    # Error file handler (errors only)
    error_handler = RotatingFileHandler(
        filename=log_dir / f"{name}.errors.log",
        maxBytes=max_size,
        backupCount=3,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(StructuredFormatter())
    logger.addHandler(error_handler)

    return logger

class ContextLogger:
    """Thread-safe logger with context binding"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.local = threading.local()
        self.local.context = {}

    def bind(self, **kwargs):
        """Add context to subsequent log messages"""
        self.local.context.update(kwargs)

    def clear_context(self):
        """Clear logging context"""
        self.local.context.clear()

    def _log(self, level: int, msg: str, **kwargs):
        """Core logging method with context"""
        extra = kwargs.pop('extra', {})
        extra['structured_data'] = {**self.local.context}
        
        self.logger.log(
            level,
            msg,
            extra=extra,
            **kwargs
        )

    def debug(self, msg: str, **kwargs):
        self._log(logging.DEBUG, msg, **kwargs)

    def info(self, msg: str, **kwargs):
        self._log(logging.INFO, msg, **kwargs)

    def warning(self, msg: str, **kwargs):
        self._log(logging.WARNING, msg, **kwargs)

    def error(self, msg: str, **kwargs):
        self._log(logging.ERROR, msg, **kwargs)

    def critical(self, msg: str, **kwargs):
        self._log(logging.CRITICAL, msg, **kwargs)

def get_logger(name: str) -> ContextLogger:
    """Get configured logger instance"""
    logger = setup_logger(name)
    return ContextLogger(name)

# Example usage
if __name__ == "__main__":
    log = get_logger("test")
    
    # Regular log
    log.info("System initialized")
    
    # Log with context
    log.bind(user="admin", subsystem="auth")
    log.warning("Weak password detected")
    
    # Error with stack trace
    try:
        1 / 0
    except Exception as e:
        log.error("Calculation failed", exc_info=True)
