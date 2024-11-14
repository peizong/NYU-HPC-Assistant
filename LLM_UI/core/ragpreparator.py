from datetime import datetime
import os
import re
import pandas as pd
import os.path

# Configuration constants
DEFAULT_CHUNK_SIZE = 1000  # Number of characters per chunk
COMPLETION_FILE_SUFFIX = '.complete'  # Suffix for completion marker file
SENTENCE_SPLIT_PATTERN = r'(?<=[.!?])\s+'  # Regex pattern for sentence splitting
REQUIRED_COLUMNS = ['content', 'file']  # Required columns in the input CSV

class RAGPreparator:
    def __init__(self, cleaned_data_file, output_file, chunk_size=DEFAULT_CHUNK_SIZE):
        self.cleaned_data_file = cleaned_data_file
        self.output_file = output_file
        self.chunk_size = chunk_size
        self.preparation_complete_file = f"{self.output_file}{COMPLETION_FILE_SUFFIX}"

    def prepare_for_rag(self):
        if os.path.exists(self.output_file) and not os.path.exists(self.preparation_complete_file):
            print("Found incomplete RAG preparation results. Deleting and starting over...")
            os.remove(self.output_file)
            
        try:
            df = pd.read_csv(self.cleaned_data_file)
            
            rag_data = []

            for idx, row in df.iterrows():
                content = row['content']
                chunks = []
                current_chunk = ""
                
                sentences = re.split(SENTENCE_SPLIT_PATTERN, content)
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                        chunks.append(current_chunk.strip())
                        current_chunk = sentence
                    else:
                        current_chunk += " " + sentence
                
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                for chunk_num, chunk in enumerate(chunks):
                    rag_data.append({
                        'file': row['file'],
                        'chunk_id': chunk_num, 
                        'chunk': chunk
                    })
            
            rag_df = pd.DataFrame(rag_data)
            rag_df.to_csv(self.output_file, index=False)
            
            # Mark preparation as complete
            with open(self.preparation_complete_file, 'w') as f:
                f.write(datetime.now().isoformat())
                
            print(f"RAG-prepared data saved to {self.output_file}")
            
        except Exception as e:
            print(f"Error during RAG preparation: {str(e)}")
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
            raise