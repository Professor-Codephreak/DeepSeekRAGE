# rage.py (c) 2025 Gregory L. Magnusson MIT license

import sys
from pathlib import Path
import psutil
import streamlit as st
from typing import Optional
from src.locallama import OllamaHandler, OllamaResponse
from src.memory import (
    memory_manager,
    ContextEntry,
    store_conversation,
    ContextType
)
from src.logger import get_logger
from src.openmind import OpenMind

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
            "temperature": 0.3,
            "streaming": False,
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

    def display_logo(self):
        """Display the RAGE logo in the sidebar."""
        with st.sidebar:
            st.image("gfx/rage_logo.png", width=200)

    def input_widget(self):
        """
        A custom input widget pinned at the bottom.
        Helper buttons appear on the right side.
        """
        # Start of bottom input container
        st.markdown('<div class="input-container">', unsafe_allow_html=True)

        # Text input on the left
        prompt = st.text_input(
            "DeepSeek with RAGE...",
            key="input_field",
            label_visibility="collapsed",
            placeholder="Enter your query...",
        )

        # Helper buttons on the right
        st.markdown(
            """
            <div class="button-group">
                <button class="stButton" title="Upload files" type="button">📁</button>
                <button class="stButton" title="Stop process" type="button">⏹️</button>
                <button class="stButton" title="Search" type="button">🔍</button>
            </div>
            """,
            unsafe_allow_html=True
        )

        # End of bottom input container
        st.markdown('</div>', unsafe_allow_html=True)

        return prompt

    def setup_sidebar(self):
        """Configure sidebar elements."""
        with st.sidebar:
            self.display_logo()
            st.header("Configuration")
            
            # Check Ollama status
            ollama_running, models = self.check_ollama_status()
            if ollama_running and models:
                st.session_state.selected_model = st.selectbox(
                    "Select Model",
                    options=models,
                    index=0
                )
            
            # Temperature control
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.01
            )
            
            # Streaming toggle
            st.session_state.streaming = st.toggle(
                "Enable Streaming",
                value=st.session_state.streaming
            )

            # Diagnostics
            cpu = psutil.cpu_percent()
            ram = psutil.virtual_memory().percent
            st.markdown(f"**CPU:** {cpu}% | **RAM:** {ram}%")

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
        if not prompt or not st.session_state.process_running:
            return
            
        try:
            model = self.initialize_ollama()
            if not model:
                return

            model.set_temperature(st.session_state.temperature)
            st.session_state.messages.append({"role": "user", "content": prompt})

            # Show user message
            with st.chat_message("user"):
                st.markdown(prompt)

            # Generate & show assistant response
            with st.chat_message("assistant"):
                with st.spinner("Processing with RAGE..."):
                    # Retrieve relevant context
                    context = self.memory.get_relevant_context(prompt)

                    # Build final prompt
                    user_prompt = self.openmind.get_user_prompt().format(
                        query=prompt,
                        context=context
                    )
                    full_prompt = (
                        f"{self.openmind.get_system_prompt()}\n\n{user_prompt}"
                    )

                    # Generate response
                    response = model.generate_response(full_prompt)
                    response_text = response.response if isinstance(response, OllamaResponse) else response

                    # Store the Q&A in conversation memory
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

                    # Display the assistant's response
                    if st.session_state.streaming:
                        response_placeholder = st.empty()
                        for chunk in response_text:
                            response_placeholder.markdown(chunk)
                    else:
                        st.markdown(response_text)

                    # Save assistant message in session
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response_text
                    })

        except Exception as e:
            logger.error(f"Processing error: {e}")
            st.error(f"Processing error: {str(e)}")
        finally:
            st.session_state.process_running = False

    def run(self):
        """Main application flow."""
        self.setup_sidebar()
        
        # Display chat messages from top to bottom
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        st.markdown('</div>', unsafe_allow_html=True)

        # Fixed input at the bottom
        if prompt := self.input_widget():
            st.session_state.process_running = True
            self.process_message(prompt)

def main():
    RAGE().run()

if __name__ == "__main__":
    main()
