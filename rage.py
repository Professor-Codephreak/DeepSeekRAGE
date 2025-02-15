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

# Initialize logger
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
            "temperature": 0.7,  # Default temperature
            "streaming": True,   # Default streaming enabled
        }
        for var, default in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default

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

    def load_css(self):
        """Load custom CSS with input field styling."""
        st.markdown("""
            <style>
            .input-container {
                display: flex;
                position: fixed;
                bottom: 2rem;
                width: 65%;
                background: var(--background-color);
                padding: 0.5rem;
                border-radius: 0.5rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                z-index: 100;
            }
            .chat-input {
                flex-grow: 1;
                margin-right: 0.5rem;
            }
            .diagnostics-box {
                position: fixed;
                top: 10px;
                right: 10px;
                padding: 10px;
                background: rgba(0, 0, 0, 0.8);
                border-radius: 5px;
                color: white;
                z-index: 1000;
            }
            .logo-container {
                text-align: center;
                margin-bottom: 1.5rem;
            }
            </style>
        """, unsafe_allow_html=True)

    def display_logo(self):
        """Display RAGE logo in sidebar."""
        with st.sidebar:
            st.markdown('<div class="logo-container">', unsafe_allow_html=True)
            st.image("gfx/rage_logo.png", width=200)
            st.markdown('</div>', unsafe_allow_html=True)

    def input_widget(self):
        """Custom input widget with integrated buttons."""
        container = st.container()
        with container:
            st.markdown('<div class="input-container">', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns([8, 1, 1])
            with col1:
                prompt = st.text_input(
                    "DeepSeek with RAGE...", 
                    key="input_field",
                    label_visibility="collapsed",
                    placeholder="Enter your query..."
                )
            with col2:
                if st.button("⏹️", help="Stop current process"):
                    st.session_state.process_running = False
                    st.rerun()
            with col3:
                if st.button("🔍", help="Search knowledge base"):
                    st.session_state.show_search = True
            
            st.markdown('</div>', unsafe_allow_html=True)
        return prompt

    def setup_sidebar(self):
        """Configure sidebar elements."""
        with st.sidebar:
            self.display_logo()
            st.header("Configuration")
            
            # Model selection and status checks
            ollama_running, models = self.check_ollama_status()
            if ollama_running:
                st.session_state.selected_model = st.selectbox(
                    "Select Model",
                    options=models,
                    index=0
                )
            
            # Temperature slider
            st.session_state.temperature = st.slider(
                "Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.temperature,
                step=0.1,
                help="Controls the randomness of the model's responses."
            )
            
            # Streaming toggle
            st.session_state.streaming = st.toggle(
                "Enable Streaming",
                value=st.session_state.streaming,
                help="Stream responses in real-time."
            )
            
            # System diagnostics
            self.display_diagnostics()

    def display_diagnostics(self):
        """Show real-time system metrics."""
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        st.markdown(f"""
            <div class="diagnostics-box">
                <strong>System Health</strong><br>
                CPU: {cpu}%<br>
                RAM: {ram}%
            </div>
        """, unsafe_allow_html=True)

    def process_message(self, prompt: str):
        """Handle message processing with interrupt support."""
        if not prompt or not st.session_state.process_running:
            return
            
        try:
            model = self.initialize_ollama()
            if not model:
                return
            
            # Add message to session state
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("Processing with RAGE..."):
                    try:
                        # Get relevant context
                        context = self.memory.get_relevant_context(prompt)
                        
                        # Format the input using the user prompt template
                        user_prompt = self.openmind.get_user_prompt().format(
                            query=prompt,
                            context=context
                        )
                        
                        # Combine system prompt and formatted user prompt
                        full_prompt = (
                            f"{self.openmind.get_system_prompt()}\n\n"
                            f"{user_prompt}"
                        )
                        
                        # Generate response
                        response = model.generate_response(
                            full_prompt,
                            temperature=st.session_state.temperature,
                            stream=st.session_state.streaming
                        )
                        
                        if isinstance(model, OllamaHandler) and model.get_last_error():
                            st.error(model.get_last_error())
                            return
                        
                        # Extract response text
                        response_text = response.response if isinstance(response, OllamaResponse) else response
                        
                        # Store conversation using ContextEntry
                        context_entry = ContextEntry(
                            content=f"Q: {prompt}\nA: {response_text}",
                            context_type=ContextType.CONVERSATION,
                            source="user",
                            metadata={
                                "provider": st.session_state.provider,
                                "model": st.session_state.selected_model,
                                "context": context
                            }
                        )
                        store_conversation(context_entry)
                        
                        # Display response
                        if st.session_state.streaming:
                            response_placeholder = st.empty()
                            for chunk in response_text:
                                response_placeholder.markdown(chunk)
                        else:
                            st.markdown(response_text)
                        
                        # Add assistant message to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                        
                    except Exception as e:
                        logger.error(f"Error generating response: {e}")
                        st.error(f"Error generating response: {str(e)}")
        except InterruptedError:
            st.warning("Process stopped by user")
        finally:
            st.session_state.process_running = False

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

    def run(self):
        """Main application flow."""
        self.setup_sidebar()
        
        # Chat history display at the top
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # Input widget at the bottom
        if prompt := self.input_widget():
            st.session_state.process_running = True
            self.process_message(prompt)

def main():
    RAGE().run()

if __name__ == "__main__":
    main()
