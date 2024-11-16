import os
import logging
import requests
from xml.etree import ElementTree
import streamlit as st
from core.faisembedder import FaissEmbedder
import json
import pandas as pd
import faiss
import pickle
import numpy as np
from openai import OpenAI

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration Variables
MODEL_NAME = "gpt-4o"  # OpenAI model to use
PAGE_TITLE = "NYU HPC Assistant"
PAGE_ICON = "🤖"
WELCOME_MESSAGE = "Ask any questions about NYU's High Performance Computing resources!"
CHAT_PLACEHOLDER = "What would you like to know about NYU's HPC?"
RESULTS_COUNT = 4
MAX_CHAT_HISTORY = 6

# Resource Configuration
RESOURCES_DIR_NAME = "resources"
RAG_DATA_FILE = "rag_prepared_data_nyu_hpc.csv"
FAISS_INDEX_FILE = "faiss_index.pkl"
S3_RESOURCES_URL = "https://nyu-hpc-llm.s3.us-east-1.amazonaws.com/"

class JinaEmbedder:
    """Handles embeddings using Jina API"""
    def __init__(self, api_key):
        self.url = "https://api.jina.ai/v1/embeddings"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }

    def get_embedding(self, text):
        data = {
            "model": "jina-embeddings-v3",
            "input": [text]
        }
        response = requests.post(self.url, headers=self.headers, json=data)
        response.raise_for_status()
        return np.array(response.json()["data"][0]["embedding"])

class FaissEmbedder:
    def __init__(self, rag_output, index_file=None):
        self.df = pd.read_csv(rag_output)
        self.embedder = JinaEmbedder(os.getenv("JINA_API_KEY"))
        self.openai_client = OpenAI()

        # Initialize index
        if index_file and os.path.exists(index_file):
            try:
                with open(index_file, 'rb') as f:
                    loaded_index = pickle.load(f)
                if isinstance(loaded_index, faiss.IndexFlatL2):
                    self.index = loaded_index
                else:
                    logger.warning("Loaded index is not a valid FAISS index. Creating new index.")
                    self._create_new_index()
            except Exception as e:
                logger.error(f"Error loading index: {str(e)}. Creating new index.")
                self._create_new_index()
        else:
            self._create_new_index()
            if index_file:
                with open(index_file, 'wb') as f:
                    pickle.dump(self.index, f)

    def _create_new_index(self):
        """Create a new FAISS index from embeddings"""
        embeddings = []
        for chunk in self.df['chunk']:
            embedding = self._get_embedding(chunk)
            embeddings.append(embedding)
        
        dimension = len(embeddings[0])
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(np.array(embeddings).astype('float32'))

    def _get_embedding(self, text):
        return self.embedder.get_embedding(text)
    
    def search(self, query, k=4):
        query_embedding = self._get_embedding(query)
        query_embedding = query_embedding.reshape(1, -1)
        D, I = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for idx in I[0]:
            results.append({
                'metadata': {
                    'chunk': self.df.iloc[idx]['chunk'],
                    'source': self.df.iloc[idx]['source']
                }
            })
        return results

def download_resources(resources_dir: str):
    """Download required resources from S3"""
    logger.info("Checking and downloading resources from S3...")
    
    try:
        # Create resources directory
        os.makedirs(resources_dir, exist_ok=True)
        
        required_files = [RAG_DATA_FILE, FAISS_INDEX_FILE]
        
        for filename in required_files:
            local_path = os.path.join(resources_dir, filename)
            
            if os.path.exists(local_path):
                logger.info(f"{filename} already exists, skipping download")
                continue
                
            file_url = S3_RESOURCES_URL + f"resources/{filename}"
            logger.info(f"Downloading {filename}...")
            
            try:
                file_response = requests.get(file_url, stream=True)
                file_response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                logger.info(f"Successfully downloaded {filename}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"Failed to download {filename}: {str(e)}")
                raise
                
    except Exception as e:
        logger.error(f"Failed to download resources: {str(e)}")
        raise

def initialize_embedder():
    """Initialize the FAISS embedder with required resources"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.join(script_dir, RESOURCES_DIR_NAME)
    
    # Check if resources exist, download if needed
    required_files = [RAG_DATA_FILE, FAISS_INDEX_FILE]
    resources_exist = all(
        os.path.exists(os.path.join(resources_dir, file))
        for file in required_files
    )
    
    if not resources_exist:
        with st.spinner('Downloading required resources...'):
            download_resources(resources_dir)
    
    rag_output = os.path.join(resources_dir, RAG_DATA_FILE)
    faiss_index_file = os.path.join(resources_dir, FAISS_INDEX_FILE)
    
    return FaissEmbedder(rag_output, index_file=faiss_index_file)

def main():
    st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON)
    
    st.title(PAGE_TITLE)
    st.markdown(WELCOME_MESSAGE)

    # Add clear chat button
    if st.sidebar.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    if "embedder" not in st.session_state:
        st.session_state.embedder = initialize_embedder()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input(CHAT_PLACEHOLDER):
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            results = st.session_state.embedder.search(prompt, k=RESULTS_COUNT)
            context = "\n".join([result['metadata']['chunk'] for result in results])
            
            chat_history = ""
            if len(st.session_state.messages) > 0:
                recent_messages = st.session_state.messages[-MAX_CHAT_HISTORY:]
                chat_history = "\nRecent conversation:\n" + "\n".join([
                    f"{msg['role']}: {msg['content']}" 
                    for msg in recent_messages
                ])
            
            messages = [
                {"role": "system", "content": """You are a helpful assistant specializing in NYU's High Performance Computing. 
First evaluate if the provided context contains relevant information for the question:
- If the context is relevant, prioritize this NYU-specific information in your response
- If the context is irrelevant or only tangentially related, rely on your general knowledge instead

Supplement your responses with general knowledge about HPC concepts, best practices, and technical explanations where appropriate.
Always ensure your responses are accurate and aligned with NYU's HPC environment."""},
                {"role": "user", "content": f"Context: {context}\n{chat_history}\n\nQuestion: {prompt}"}
            ]
            
            # Stream the response
            stream = st.session_state.embedder.openai_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                stream=True,
            )

            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    full_response += chunk.choices[0].delta.content
                    message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
        
        st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
