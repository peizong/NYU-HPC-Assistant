import os
import pandas as pd
from tqdm import tqdm
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle
import faiss
import os.path
from openai import OpenAI

# Configuration Constants
DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
DEFAULT_LLM_MODEL = "gpt-4o-mini" # THIS IS NOT THE MODEL USED IN THE STREAMLIT APP
DEFAULT_INDEX_FILE = "faiss_index.pkl"
DEFAULT_CHECKPOINT_SUFFIX = "embedding_checkpoint.json"
DEFAULT_SEARCH_RESULTS = 5
CHECKPOINT_SAVE_FREQUENCY = 5  # Save checkpoint every N iterations

# System prompt for the LLM
SYSTEM_PROMPT = """You are a helpful assistant specializing in NYU's High Performance Computing.
First evaluate if the provided context contains relevant information for the question:
- If the context is relevant, prioritize this NYU-specific information in your response
- If the context is irrelevant or only tangentially related, rely on your general knowledge instead

Supplement your responses with general knowledge about HPC concepts, best practices, and technical explanations where appropriate.
Always ensure your responses are accurate and aligned with NYU's HPC environment."""

class FaissEmbedder:
    def __init__(self, rag_data_file, index_file=DEFAULT_INDEX_FILE, checkpoint_file=None, openai_client=None): # Added openai_client parameter
        self.rag_data_file = rag_data_file
        self.index_file = index_file
        # Store checkpoint in same directory as index file
        self.checkpoint_file = checkpoint_file or os.path.join(os.path.dirname(index_file), DEFAULT_CHECKPOINT_SUFFIX)
        self.model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL, trust_remote_code=True)
        self.dimension = self.model.get_sentence_embedding_dimension()
        if openai_client: # Use provided client if available
            self.openai_client = openai_client
        else:
            self.openai_client = OpenAI() # Default client if none provided

    def load_checkpoint(self):
        try:
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {'last_processed_index': -1}

    def save_checkpoint(self, last_index):
        with open(self.checkpoint_file, 'w') as f:
            json.dump({'last_processed_index': last_index}, f)

    def create_index(self):
        index = faiss.IndexFlatL2(self.dimension)
        return index

    def embed_and_insert(self):
        df = pd.read_csv(self.rag_data_file)
        checkpoint = self.load_checkpoint()
        start_index = checkpoint['last_processed_index'] + 1

        # If we have a partial index, load it
        if start_index > 0 and os.path.exists(self.index_file):
            with open(self.index_file, 'rb') as f:
                data = pickle.load(f)
                index = data['index']
                metadata = data['metadata']
        else:
            index = self.create_index()
            metadata = []

        try:
            # Use enumerate to keep track of absolute position in DataFrame
            for idx, (_, row) in tqdm(enumerate(df.iloc[start_index:].iterrows(), start=start_index),
                                    total=len(df) - start_index,  # Adjust total for correct progress bar
                                    initial=start_index,
                                    desc="Embedding and inserting"):
                # Skip if chunk is empty or just whitespace
                if not row['chunk'] or not str(row['chunk']).strip():
                    continue

                embedding = self.model.encode(row['chunk'])
                index.add(np.array([embedding]))
                metadata.append({
                    'file': row['file'],
                    'chunk_id': row['chunk_id'],
                    'chunk': row['chunk']
                })

                # Save checkpoint periodically
                if idx % 5 == 0:
                    self.save_checkpoint(idx)
                    with open(self.index_file, 'wb') as f:
                        pickle.dump({'index': index, 'metadata': metadata}, f)

            # Final save
            with open(self.index_file, 'wb') as f:
                pickle.dump({'index': index, 'metadata': metadata}, f)
            # Clear checkpoint after successful completion
            if os.path.exists(self.checkpoint_file):
                os.remove(self.checkpoint_file)

        except Exception as e:
            print(f"Embedding process interrupted: {str(e)}")
            # Save progress before raising the exception
            self.save_checkpoint(idx)
            with open(self.index_file, 'wb') as f:
                pickle.dump({'index': index, 'metadata': metadata}, f)
            raise

        print(f"Inserted {index.ntotal} entities into FAISS index and saved to {self.index_file}")

    def search(self, query, k=DEFAULT_SEARCH_RESULTS):
        # Load the index and metadata
        with open(self.index_file, 'rb') as f:
            data = pickle.load(f)
            index = data['index']
            metadata = data['metadata']

        # Encode query
        query_vector = self.model.encode(query)

        # Perform search
        distances, indices = index.search(np.array([query_vector]), k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'distance': distances[0][i],
                'metadata': metadata[idx]
            })

        return results

    def generate_answer(self, query, k=DEFAULT_SEARCH_RESULTS):
        results = self.search(query, k=k)
        context = "\n".join([result['metadata']['chunk'] for result in results])

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]

        stream = self.openai_client.chat.completions.create( # Using self.openai_client
            model=DEFAULT_LLM_MODEL,
            messages=messages,
            stream=True,
        )

        print("\nAnswer:")
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print("\n")

    def interactive_search_and_answer(self):
        print("Welcome to the NYU HPC Search and Answer. Type 'quit' to exit.")
        while True:
            query = input("\nEnter your question: ")
            if query.lower() == 'quit':
                print("Thank you for using NYU HPC Search and Answer. Goodbye!")
                break

            self.generate_answer(query)

            # print("\nTop 3 relevant chunks:")
            # results = self.search(query, k=3)
            # for i, result in enumerate(results, 1):
            #     print(f"\n{i}. Relevance: {1 / (1 + result['distance']):.2f}")
            #     print(f"Chunk: {result['metadata']['chunk'][:200]}...")
            #     print(f"Source: {result['metadata']['file']}")
