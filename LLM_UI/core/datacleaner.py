
import os
import pandas as pd
import re
from tqdm import tqdm
import trafilatura
from datetime import datetime
import os.path

class DataCleaner:
    def __init__(self, input_folder, output_file):
        self.input_folder = input_folder
        self.output_file = output_file
        self.cleaning_complete_file = f"{self.output_file}.complete"
    
    @staticmethod
    def extract_main_content(html_content):
            
        extracted = trafilatura.extract(html_content, include_links=False, include_images=False, include_tables=False)
        if extracted:
            cleaned = re.sub(r'\s+', ' ', extracted).strip() 
            cleaned = re.sub(r'\n+', '\n', cleaned)  
            return cleaned
        return None

    def clean_data(self):
        # Delete partial results if they exist
        if os.path.exists(self.output_file) and not os.path.exists(self.cleaning_complete_file):
            print("Found incomplete cleaning results. Deleting and starting over...")
            os.remove(self.output_file)
        
        try:
            data = []
            
            for root, dirs, files in os.walk(self.input_folder):
                for file in tqdm(files, desc="Cleaning data"):
                    if file.endswith('.html'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                
                                # Extract main content
                                main_content = self.extract_main_content(content)
                                
                                if main_content:
                                    data.append({
                                        'file': file_path,
                                        'content': main_content
                                    })
                                else:
                                    print(f"No main content extracted from {file_path}")
                        except Exception as e:
                            print(f"Error processing {file_path}: {str(e)}")
            
            df = pd.DataFrame(data)
            df.to_csv(self.output_file, index=False)
            
            # Mark cleaning as complete
            with open(self.cleaning_complete_file, 'w') as f:
                f.write(datetime.now().isoformat())
                
            print(f"Cleaned data saved to {self.output_file}")
            
        except Exception as e:
            print(f"Error during cleaning: {str(e)}")
            # Delete partial results
            if os.path.exists(self.output_file):
                os.remove(self.output_file)
            raise
