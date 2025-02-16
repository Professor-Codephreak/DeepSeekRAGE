# rage.py (c) 2025 Gregory L. Magnusson MIT license

import sys
import json
import time
import psutil
import streamlit as st
from pathlib import Path
from typing import Optional
from src.logger import get_logger
from src.openmind import OpenMind
from src.locallama import OllamaHandler, OllamaResponse
from src.memory import (
    memory_manager,
    ContextEntry,
    store_conversation,
    ContextType
)

# Set the favicon and page title
st.set_page_config(
    page_title="RAGE",          # Title of the page
    page_icon="gfx/rage.ico",   # Path to the favicon file
    layout="wide"               # Optional: Set layout to "wide" or "centered"
)


logger = get_logger('rage')

class RAGE:
    """RAGE - Retrieval Augmented Generative Engine (DeepSeeker)"""
    
    def __init__(self):
        self.setup_session_state()
        self.load_css()
        self.memory = memory_manager
        self.openmind = OpenMind()

    def setup_session_state(self):
        """Initialize session state variables."""
        session_vars = {
            "messages": [],
            "provider": "Ollama",
            "selected_model": None,
            "model_instances": {'ollama': None},
            "process_running": False,
            "show_search": False,
            "temperature": 0.30,
            "streaming": False,
            "current_response": "",
        }
        for var, default in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default

    def load_css(self):
        """Load external CSS from 'gfx/styles.css'."""
        try:
            with open("gfx/styles.css", "r") as f:
                css = f.read()
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        except FileNotFoundError:
            st.error("Could not find 'gfx/styles.css'. Please ensure it exists.")

    def display_diagnostics(self):
        """Display system diagnostics in top right corner."""
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        st.markdown(
            f'<div class="diagnostics-box">CPU: {cpu}% | RAM: {ram}%</div>',
            unsafe_allow_html=True
        )

    def setup_sidebar(self):
        """Configure sidebar elements."""
        with st.sidebar:
            # Configuration Header
            st.markdown("### Configuration")
            
            # Model Selection
            st.markdown("#### Select Model")
            ollama_running, models = self.check_ollama_status()
            if ollama_running and models:
                st.session_state.selected_model = st.selectbox(
                    "Model Selection",
                    options=models,
                    index=0 if models else None,
                    label_visibility="collapsed"
                )
            
            # Temperature Slider
            st.markdown("#### Temperature")
            st.session_state.temperature = st.slider(
                "Temperature Control",
                min_value=0.00,
                max_value=1.00,
                value=st.session_state.temperature,
                step=0.01,
                label_visibility="collapsed",
                format="%.2f"
            )
            
            # Streaming Toggle
            st.session_state.streaming = st.toggle(
                "Enable Streaming",
                value=st.session_state.streaming,
                key="streaming_toggle"
            )

            # Conversation History
            if st.session_state.messages:
                st.markdown("### Conversation History")
                for msg in st.session_state.messages:
                    if msg["role"] == "user":
                        truncated_content = msg["content"][:50]
                        if len(msg["content"]) > 50:
                            truncated_content += "..."
                        st.markdown(
                            f'<div class="chat-history-item">{truncated_content}</div>',
                            unsafe_allow_html=True
                        )

    def check_ollama_status(self):
        """Check Ollama installation and available models."""
        try:
            if not st.session_state.model_instances['ollama']:
                st.session_state.model_instances['ollama'] = OllamaHandler()
            
            if st.session_state.model_instances['ollama'].check_installation():
                models = st.session_state.model_instances['ollama'].list_models()
                return True, models
            return False, []
        except Exception as e:
            logger.error(f"Error checking Ollama status: {e}")
            return False, []

    def initialize_ollama(self) -> Optional[OllamaHandler]:
        """Initialize or retrieve Ollama model instance."""
        try:
            if not st.session_state.model_instances['ollama']:
                st.session_state.model_instances['ollama'] = OllamaHandler()
            
            if st.session_state.model_instances['ollama'].check_installation():
                available_models = st.session_state.model_instances['ollama'].list_models()
                if available_models:
                    if not st.session_state.selected_model:
                        st.info("Please select an Ollama model to continue")
                        return None
                    
                    if st.session_state.model_instances['ollama'].select_model(st.session_state.selected_model):
                        return st.session_state.model_instances['ollama']
                    else:
                        st.error(st.session_state.model_instances['ollama'].get_last_error())
                        return None
                else:
                    st.error("No Ollama models found. Please pull a model first.")
                    return None
            else:
                st.error("Ollama service is not running. Please start the Ollama service.")
                return None
        except Exception as e:
            logger.error(f"Error initializing Ollama: {e}")
            st.error(f"Error initializing Ollama: {str(e)}")
            return None

    def process_message(self, prompt: str):
        """Process user input with RAGE engine."""
        if not prompt:
            return
                
        try:
            model = self.initialize_ollama()
            if not model:
                return

            # Set model parameters
            model.set_temperature(st.session_state.temperature)
            model.set_streaming(st.session_state.streaming)
            
            # Add user message to history
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Show user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate & show assistant response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                
                # Get context and build prompt
                context = self.memory.get_relevant_context(prompt)
                user_prompt = self.openmind.get_user_prompt().format(
                    query=prompt,
                    context=context
                )
                full_prompt = f"{self.openmind.get_system_prompt()}\n\n{user_prompt}"

                start_time = time.time()
                
                # Generate response based on streaming setting
                if st.session_state.streaming:
                    full_response = ""
                    for chunk in model.generate_response(full_prompt):
                        try:
                            if isinstance(chunk, str):
                                chunk_data = json.loads(chunk)
                                if "response" in chunk_data:
                                    full_response += chunk_data["response"]
                                    message_placeholder.markdown(full_response)
                        except json.JSONDecodeError:
                            if isinstance(chunk, str):
                                full_response += chunk
                                message_placeholder.markdown(full_response)
                    response_text = full_response
                    elapsed_time = time.time() - start_time
                    message_placeholder.markdown(f"{response_text}\n\n*Response time: {elapsed_time:.2f}s*")
                else:
                    spinner_placeholder = st.empty()
                    with spinner_placeholder:
                        with st.spinner("RAGE is thinking...", show_time=True):
                            response = model.generate_response(full_prompt)
                            response_text = response.response if isinstance(response, OllamaResponse) else str(response)
                            elapsed_time = time.time() - start_time
                            message_placeholder.markdown(f"{response_text}\n\n*Response time: {elapsed_time:.2f}s*")
                    spinner_placeholder.empty()

                # Store in memory
                if response_text:
                    store_conversation(ContextEntry(
                        content=f"Q: {prompt}\nA: {response_text}",
                        context_type=ContextType.CONVERSATION,
                        source="user",
                        metadata={
                            "provider": st.session_state.provider,
                            "model": st.session_state.selected_model,
                            "context": context
                        }
                    ))

                    # Update session messages
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"{response_text}\n\n*Response time: {elapsed_time:.2f}s*"
                    })

        except Exception as e:
            logger.error(f"Processing error: {e}")
            st.error(f"Processing error: {str(e)}")

    def run(self):
        """Main application flow."""
        # Display diagnostics
        self.display_diagnostics()
        
        # Setup sidebar
        self.setup_sidebar()
        
        # Main chat container
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Input container with inline helper buttons
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # Chat input
        prompt = st.chat_input(
            placeholder="DeepSeek with RAGE...",
            key="chat_input"
        )
        
        # Helper buttons (inline with send button)
        st.markdown(
            """
            <div class="button-group">
                <button class="stButton" title="Upload files">📁</button>
                <button class="stButton" title="Stop process">⏹️</button>
                <button class="stButton" title="Search">🔍</button>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        if prompt:
            self.process_message(prompt)


def main():
    RAGE().run()


if __name__ == "__main__":
    main()
