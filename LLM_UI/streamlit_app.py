import streamlit as st
import os
from core.faisembedder import FaissEmbedder

# Configuration Variables
MODEL_NAME = "gpt-4o"  # OpenAI model to use
PAGE_TITLE = "NYU HPC Assistant"
PAGE_ICON = "🤖"
WELCOME_MESSAGE = "Ask any questions about NYU's High Performance Computing resources!"
CHAT_PLACEHOLDER = "What would you like to know about NYU's HPC?"
RESULTS_COUNT = 4  # Number of similar documents to retrieve
MAX_CHAT_HISTORY = 6  # Number of recent messages to include in context

RESOURCES_FOLDER = "resources"
RAG_DATA_FILE = "rag_prepared_data_nyu_hpc.csv"
FAISS_INDEX_FILE = "faiss_index.pkl"

def initialize_embedder():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.join(script_dir, RESOURCES_FOLDER)
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
                {"role": "system", "content": "You are a helpful assistant specializing in NYU's High Performance Computing. When answering questions, carefully evaluate the relevance of each context chunk provided. Some chunks may be irrelevant - only use information that directly relates to the question. First prioritize relevant information from the provided context for NYU-specific details. Then supplement this with your general knowledge about HPC concepts, best practices, and technical explanations where appropriate. Always ensure your responses are accurate and aligned with NYU's HPC environment. If none of the context chunks are relevant, rely on your general knowledge while staying within the scope of NYU's HPC environment."},
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