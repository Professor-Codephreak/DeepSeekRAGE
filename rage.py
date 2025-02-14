# rage.py (c) 2025 Gregory L. Magnusson MIT license
# RAGE Retrieval Augmented Generative Engine (c) 2025 rage.pythai.net

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
from typing import Optional
from src.locallama import OllamaHandler, OllamaResponse
from src.memory import (
    memory_manager,
    DialogEntry,
    store_conversation
)
# from src.config import get_config, get_model_config  # Commented out
from src.logger import get_logger
from src.openmind import OpenMind

# Initialize logger
logger = get_logger('rage')

class RAGE:
    """RAGE - Retrieval Augmented Generative Engine (DeepSeeker)"""
    
    def __init__(self):
        self.setup_session_state()
        # self.config = get_config()  # Commented out
        # self.model_config = get_model_config()  # Commented out
        self.load_css()
        
        # Initialize systems
        self.memory = memory_manager
        self.openmind = OpenMind()
    
    def setup_session_state(self):
        """Initialize session state variables."""
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if 'provider' not in st.session_state:
            st.session_state.provider = "Ollama"  # Ollama localhost
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = None
        if 'model_instances' not in st.session_state:
            st.session_state.model_instances = {'ollama': None}
    
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
        """Load CSS styling."""
        try:
            with open('gfx/styles.css') as f:
                st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
        except Exception as e:
            logger.error(f"Error loading CSS: {e}")
            self.load_default_css()
    
    def load_default_css(self):
        """Load default CSS if custom CSS fails."""
        st.markdown("""
            <style>
            .cost-tracker { padding: 10px; background: #262730; border-radius: 5px; }
            .model-info { padding: 10px; background: #1E1E1E; border-radius: 5px; }
            .capability-tag { 
                display: inline-block; 
                padding: 2px 8px; 
                margin: 2px;
                background: #3B3B3B; 
                border-radius: 12px; 
                font-size: 0.8em; 
            }
            .api-key-status {
                display: flex;
                align-items: center;
                gap: 8px;
                padding: 5px;
                margin: 5px 0;
            }
            .checkmark {
                color: #00cc00;
                font-weight: bold;
            }
            </style>
        """, unsafe_allow_html=True)
    
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
        """Process user message and generate response using Ollama."""
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
                        
                        # Generate response
                        response = model.generate_response(prompt, context)
                        
                        if isinstance(model, OllamaHandler) and model.get_last_error():
                            st.error(model.get_last_error())
                            return
                        
                        # Extract response text if it's an OllamaResponse object
                        response_text = response.response if isinstance(response, OllamaResponse) else response
                        
                        # Store conversation
                        dialog_entry = DialogEntry(
                            query=prompt,
                            response=response_text,
                            provider=st.session_state.provider,
                            model=st.session_state.selected_model,
                            context={"retrieved_context": context}
                        )
                        store_conversation(dialog_entry)
                        
                        # Display response
                        st.markdown(response_text)
                        
                        # Add assistant message to session state
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response_text
                        })
                        
                    except Exception as e:
                        logger.error(f"Error generating response: {e}")
                        st.error(f"Error generating response: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            st.error("An error occurred while processing your message")
    
    def setup_sidebar(self):
        """Setup sidebar configuration."""
        with st.sidebar:
            st.header("RAGE Configuration")
            
            # Check Ollama status
            ollama_running, ollama_models = self.check_ollama_status()
            if ollama_running:
                st.markdown("""
                    <div class="api-key-status">
                        <span class="checkmark">●</span>
                        <span class="text">Ollama Running</span>
                    </div>
                    """, unsafe_allow_html=True)
                if ollama_models:
                    st.caption(f"Available models: {', '.join(ollama_models)}")
            
            # Model selection
            if ollama_models:
                st.session_state.selected_model = st.selectbox(
                    "Select Ollama Model",
                    options=ollama_models,
                    key='ollama_model_select'
                )
            
            # Display model information (commented out since model_config is removed)
            # if st.session_state.selected_model:
            #     model_info = self.model_config.get_model_info(
            #         st.session_state.provider.lower(),
            #         st.session_state.selected_model
            #     )
            #     if model_info:
            #         st.markdown("### Model Information")
            #         st.markdown(f"""
            #         <div class="model-info">
            #             <p><strong>Model:</strong> {model_info.name}</p>
            #             <p><strong>Developer:</strong> {model_info.developer}</p>
            #             <p><strong>Max Tokens:</strong> {model_info.tokens}</p>
            #             <p><strong>Cost:</strong> {model_info.cost}</p>
            #             <div><strong>Capabilities:</strong></div>
            #             {''.join([f'<span class="capability-tag">{cap}</span>' 
            #                     for cap in model_info.capabilities])}
            #         </div>
            #         """, unsafe_allow_html=True)
    
    def run(self):
        """Run the RAGE interface."""
        try:
            st.title("RAGE - Retrieval Augmented Generative Engine")
            
            # Setup sidebar
            self.setup_sidebar()
            
            # Chat interface
            chat_container = st.container()
            
            with chat_container:
                # Display conversation history
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("DeepSeek with RAGE..."):
                self.process_message(prompt)
            
        except Exception as e:
            logger.error(f"Main application error: {e}")
            st.error("An error occurred in the application. Please refresh the page.")

def main():
    rage = RAGE()
    rage.run()

if __name__ == "__main__":
    main()
