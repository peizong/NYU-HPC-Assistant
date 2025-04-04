
import os
from openai import OpenAI
from portkey_ai import PORTKEY_GATEWAY_URL, createHeaders
import pickle
from sentence_transformers import SentenceTransformer
import numpy as np

def search(query, k):
        # Load the index and metadata
        index_file="resources/faiss_index.pkl"
        with open(index_file, 'rb') as f:
            data = pickle.load(f)
            index = data['index']
            metadata = data['metadata']

        # Encode query
        DEFAULT_EMBEDDING_MODEL = "jinaai/jina-embeddings-v3"
        model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL, trust_remote_code=True)
        query_vector = model.encode(query)

        # Perform search
        distances, indices = index.search(np.array([query_vector]), k)

        results = []
        for i, idx in enumerate(indices[0]):
            results.append({
                'distance': distances[0][i],
                'metadata': metadata[idx]
            })

        return results

client = OpenAI(
  api_key="xxx", #os.environ.get("OPENAI_API_KEY"),
  base_url="https://ai-gateway.apps.cloud.rt.nyu.edu/v1", #PORTKEY_GATEWAY_URL, # 👈 or 'http://localhost:8787/v1'
  default_headers=createHeaders(
    provider="openai",
    api_key="8gTMTBfxZ9zzXHp/ZTcbUhPo9+81", #os.environ.get("PORTKEY_API_KEY") # 👈 skip when self-hosting
    virtual_key= "openai-nyu-it-d-5b382a"
  )
)

prompt="How to open an account?"
RESULTS_COUNT = 4
results = search(prompt, k=RESULTS_COUNT)
context = "\n".join([result['metadata']['chunk'] for result in results])
print("context: ", context)

recent_messages =[[]] #st.session_state.messages[-MAX_CHAT_HISTORY:]
chat_history = ""
# chat_history = "\nRecent conversation:\n" + "\n".join([
#                     f"{msg['role']}: {msg['content']}" 
#                     for msg in recent_messages
#                 ])

messages = [
                {"role": "system", "content": """You are a helpful assistant specializing in NYU's High Performance Computing. 
First evaluate if the provided context contains relevant information for the question:
- If the context is relevant, prioritize this NYU-specific information in your response.
- If the context is irrelevant or only tangentially related, rely on your general knowledge instead.
- Do not mention "context", the user does not know how the code works internally.

Supplement your responses with general knowledge about HPC concepts, best practices, and technical explanations where appropriate.
Always ensure your responses are accurate and aligned with NYU's HPC environment."""},
                {"role": "user", "content": f"Context: {context}\n{chat_history}\n\nQuestion: {prompt}"}]

stream=client.chat.completions.create(
  model="gpt-4o-mini",
  messages=messages, #[{"role": "user", "content": "What is a fractal?"}],
  #stream=True,
)

print(stream.choices[0].message.content) #.choices[0].message.content)

full_response = ""

for chunk in [stream]:
                if chunk.choices[0].message.content is not None:
                    full_response += chunk.choices[0].message.content
                    #message_placeholder.markdown(full_response + "▌")


