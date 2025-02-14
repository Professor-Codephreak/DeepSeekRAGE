# src/locallama.py (c) 2025 Gregory L. Magnusson MIT License

from typing import Optional, List, Dict, Any, Generator, Union
import requests
import logging
from dataclasses import dataclass

# Initialize logger
logger = logging.getLogger('locallama')

# Default Deepseek models
DEFAULT_MODELS = ["deepseek-r1:1.5b", "deepseek-r1:8b", "deepseek-r1:14b"]

@dataclass
class OllamaResponse:
    """Dataclass to structure Ollama API responses"""
    response: str
    model: str
    created_at: str
    done: bool
    context: Optional[List[int]] = None
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    prompt_eval_count: Optional[int] = None
    prompt_eval_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

class OllamaHandler:
    """Handler for interacting with the Ollama API"""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.last_error: Optional[str] = None
        self.selected_model: Optional[str] = None
        self.temperature: float = 0.314  # Default temperature
        self.streaming: bool = False  # Default streaming off

    def check_installation(self) -> bool:
        """Check if Ollama service is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                return True
            self.last_error = f"Ollama service not running: {response.text}"
            return False
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Ollama installation check failed: {e}")
            return False

    def list_models(self) -> List[str]:
        """List available Ollama models using 'ollama list' equivalent"""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                return [model["name"] for model in models]
            self.last_error = f"Failed to list models: {response.text}"
            return []
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error listing models: {e}")
            return []

    def select_model(self, model_name: str) -> bool:
        """Select Ollama model using 'ollama run modelname' equivalent"""
        try:
            # Check if model exists
            models = self.list_models()
            if model_name not in models:
                # Try to pull the model using 'ollama pull' equivalent
                response = requests.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name}
                )
                if response.status_code != 200:
                    self.last_error = f"Failed to pull model {model_name}: {response.text}"
                    return False

                # Verify the model was pulled successfully
                models = self.list_models()
                if model_name not in models:
                    self.last_error = f"Model '{model_name}' not found after pulling"
                    return False

            # Set the selected model
            self.selected_model = model_name
            return True

        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error selecting model: {e}")
            return False

    def set_temperature(self, temperature: float):
        """Set the temperature for response generation (1/1000 precision)"""
        if 0.0 <= temperature <= 1.0:
            self.temperature = round(temperature, 3)  # Round to 1/1000 precision
        else:
            self.last_error = "Temperature must be between 0.0 and 1.0"

    def set_streaming(self, streaming: bool):
        """Enable or disable streaming responses"""
        self.streaming = streaming

    def generate_response(
        self, prompt: str, context: Optional[str] = None
    ) -> Optional[Union[OllamaResponse, Generator[str, None, None]]]:
        """Generate a response using the selected model"""
        if not self.selected_model:
            self.last_error = "No model selected"
            return None

        payload = {
            "model": self.selected_model,
            "prompt": prompt,
            "stream": self.streaming,
            "temperature": self.temperature,
        }
        if context:
            payload["context"] = context

        try:
            if self.streaming:
                # Handle streaming response
                response = requests.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                    stream=True
                )
                if response.status_code == 200:
                    return self._handle_streaming_response(response)
                else:
                    self.last_error = f"API Error: {response.text}"
                    return None
            else:
                # Handle non-streaming response
                response = requests.post(f"{self.base_url}/api/generate", json=payload)
                if response.status_code == 200:
                    response_data = response.json()
                    return OllamaResponse(
                        response=response_data.get("response", ""),
                        model=response_data.get("model", ""),
                        created_at=response_data.get("created_at", ""),
                        done=response_data.get("done", False),
                        context=response_data.get("context"),
                        total_duration=response_data.get("total_duration"),
                        load_duration=response_data.get("load_duration"),
                        prompt_eval_count=response_data.get("prompt_eval_count"),
                        prompt_eval_duration=response_data.get("prompt_eval_duration"),
                        eval_count=response_data.get("eval_count"),
                        eval_duration=response_data.get("eval_duration")
                    )
                else:
                    self.last_error = f"API Error: {response.text}"
                    return None
        except Exception as e:
            self.last_error = f"Error during generation: {e}"
            return None

    def _handle_streaming_response(self, response) -> Generator[str, None, None]:
        """Handle streaming response from Ollama"""
        try:
            for chunk in response.iter_content(chunk_size=None):
                if chunk:
                    yield chunk.decode("utf-8")
        except Exception as e:
            self.last_error = f"Error during streaming: {e}"
            yield f"Error: {str(e)}"

    def get_last_error(self) -> Optional[str]:
        """Get the last error message"""
        return self.last_error

    def get_embedding(self, text: str, model: Optional[str] = None) -> Optional[List[float]]:
        """Get embeddings for a given text"""
        model_to_use = model or self.selected_model
        if not model_to_use:
            self.last_error = "No model selected for embedding"
            return None

        payload = {
            "model": model_to_use,
            "prompt": text
        }

        try:
            response = requests.post(f"{self.base_url}/api/embeddings", json=payload)
            if response.status_code == 200:
                return response.json().get("embedding")
            self.last_error = f"Error getting embedding: {response.text}"
            return None
        except Exception as e:
            self.last_error = f"Error during embedding generation: {e}"
            return None

    def auto_select_default_model(self) -> bool:
        """Auto-select the first available Deepseek model"""
        try:
            models = self.list_models()
            for model in DEFAULT_MODELS:
                if model in models:
                    self.selected_model = model
                    return True
            self.last_error = "No default Deepseek models found"
            return False
        except Exception as e:
            self.last_error = str(e)
            logger.error(f"Error auto-selecting default model: {e}")
            return False
