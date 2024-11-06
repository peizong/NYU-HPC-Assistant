import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import pandas as pd
import re
import time
import random
import logging
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import trafilatura
import json
from datetime import datetime
import numpy as np
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from sentence_transformers import SentenceTransformer
import pickle
import faiss
from transformers import pipeline
import os.path
from openai import OpenAI

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class WebScraper:
    def __init__(self, base_url, output_folder, url_file='scraped_urls.json'):
        self.base_url = base_url
        self.output_folder = output_folder
        self.visited_urls = set()
        self.url_file = url_file
        self.scraped_urls = self.load_scraped_urls()
        self.use_selenium = False
        self.driver = None
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

        # Retry 
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.1, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

    def load_scraped_urls(self):
        try:
            with open(self.url_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_scraped_urls(self):
        with open(self.url_file, 'w') as f:
            json.dump(self.scraped_urls, f, indent=2)

    def get_page_content(self, url):
        time.sleep(random.uniform(2, 5))
        
        try:
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            logger.warning(f"Error fetching {url} with requests: {str(e)}. Trying with Selenium.")
            return self.get_page_content_selenium(url)

    def get_page_content_selenium(self, url):
        if not self.driver:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")
            self.driver = webdriver.Chrome(options=chrome_options)

        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            return self.driver.page_source
        except Exception as e:
            logger.error(f"Error fetching {url} with Selenium: {str(e)}")
            return None

    def save_page(self, url, content):
        parsed_url = urlparse(url)
        
        safe_path = re.sub(r'[<>:"/\\|?*]', '_', parsed_url.path.strip('/'))
        file_path = os.path.join(self.output_folder, parsed_url.netloc, safe_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(f"{file_path}.html", 'w', encoding='utf-8') as f:
            f.write(content)

    def scrape_page(self, url):
        current_time = datetime.now().isoformat()
        
        if url in self.scraped_urls:
            logger.info(f"Already scraped: {url}")
            return []
        
        self.visited_urls.add(url)
        logger.info(f"Scraping: {url}")
        
        content = self.get_page_content(url)
        
        if not content:
            logger.error(f"Failed to fetch content for {url}")
            return []

        self.save_page(url, content)
        
        # Update scraped_urls dictionary
        self.scraped_urls[url] = current_time
        self.save_scraped_urls()
        
        soup = BeautifulSoup(content, 'html.parser')
        links = soup.find_all('a', href=True)
        
        new_urls = []
        for link in links:
            new_url = urljoin(self.base_url, link['href'])
            if new_url.startswith(self.base_url) and new_url not in self.visited_urls:
                new_urls.append(new_url)
        
        time.sleep(random.uniform(1, 3))
        return new_urls

    def scrape(self):
        urls_to_scrape = [self.base_url]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            while urls_to_scrape:
                new_urls = list(executor.map(self.scrape_page, urls_to_scrape))
                urls_to_scrape = [url for sublist in new_urls for url in sublist if url not in self.scraped_urls]

        if self.driver:
            self.driver.quit()

class DataCleaner:
    def __init__(self, input_folder, output_file):
        self.input_folder = input_folder
        self.output_file = output_file
    
    @staticmethod
    def extract_main_content(html_content):
            
        extracted = trafilatura.extract(html_content, include_links=False, include_images=False, include_tables=False)
        if extracted:
            cleaned = re.sub(r'\s+', ' ', extracted).strip() 
            cleaned = re.sub(r'\n+', '\n', cleaned)  
            return cleaned
        return None

    def clean_data(self):
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
                                logger.warning(f"No main content extracted from {file_path}")
                    except Exception as e:
                        logger.error(f"Error processing {file_path}: {str(e)}")
        
        df = pd.DataFrame(data)
        df.to_csv(self.output_file, index=False)
        logger.info(f"Cleaned data saved to {self.output_file}")

class RAGPreparator:
    def __init__(self, cleaned_data_file, output_file, chunk_size=1000):
        self.cleaned_data_file = cleaned_data_file
        self.output_file = output_file
        self.chunk_size = chunk_size

    def prepare_for_rag(self):
        """
        Split the cleaned data into chunks for RAG.
        """
        df = pd.read_csv(self.cleaned_data_file)
        
        rag_data = []

        for idx, row in df.iterrows():
            content = row['content']
            chunks = []
            current_chunk = ""
            
            sentences = re.split(r'(?<=[.!?])\s+', content)
            
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
        logger.info(f"RAG-prepared data saved to {self.output_file}")

class FaissEmbedder:
    def __init__(self, rag_data_file, index_file="faiss_index.pkl"):
        self.rag_data_file = rag_data_file
        self.index_file = index_file
        self.model = SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)
        self.dimension = self.model.get_sentence_embedding_dimension()
        self.openai_client = OpenAI()

    def create_index(self):
        index = faiss.IndexFlatL2(self.dimension)
        return index

    def embed_and_insert(self):
        df = pd.read_csv(self.rag_data_file)
        index = self.create_index()
        metadata = []

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Embedding and inserting"):
            embedding = self.model.encode(row['chunk'])
            index.add(np.array([embedding]))
            metadata.append({
                'file': row['file'],
                'chunk_id': row['chunk_id'],
                'chunk': row['chunk']
            })

        # Save the index and metadata
        with open(self.index_file, 'wb') as f:
            pickle.dump({'index': index, 'metadata': metadata}, f)

        logger.info(f"Inserted {index.ntotal} entities into FAISS index and saved to {self.index_file}")

    def search(self, query, k=5):
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

    def generate_answer(self, query, k=5):
        results = self.search(query, k=k)
        context = "\n".join([result['metadata']['chunk'] for result in results])
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant for NYU's High Performance Computing. Use the following context to answer the user's question."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
        ]
        
        stream = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
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

            print("\nTop 3 relevant chunks:")
            results = self.search(query, k=3)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. Relevance: {1 / (1 + result['distance']):.2f}")
                print(f"Chunk: {result['metadata']['chunk'][:200]}...")
                print(f"Source: {result['metadata']['file']}")

if __name__ == "__main__":
    base_url = "https://sites.google.com/nyu.edu/nyu-hpc/"
    output_folder = "scraped_data_nyu_hpc"
    cleaned_output = "cleaned_data_nyu_hpc.csv"
    rag_output = "rag_prepared_data_nyu_hpc.csv"
    faiss_index_file = "faiss_index.pkl"

    # 1: Scrape website (if needed)
    scraper = WebScraper(base_url, output_folder, url_file='nyu_hpc_scraped_urls.json')
    if not os.path.exists(output_folder) or not os.listdir(output_folder):
        logger.info("Starting web scraping...")
        scraper.scrape()
    else:
        logger.info("Scraped data already exists. Skipping scraping step.")

    # 2: Clean data (if needed)
    if not os.path.exists(cleaned_output):
        logger.info("Starting data cleaning...")
        cleaner = DataCleaner(output_folder, cleaned_output)
        cleaner.clean_data()
    else:
        logger.info("Cleaned data already exists. Skipping cleaning step.")

    # 3: Prepare for RAG (if needed)
    if not os.path.exists(rag_output):
        logger.info("Starting RAG preparation...")
        preparator = RAGPreparator(cleaned_output, rag_output, chunk_size=1000)
        preparator.prepare_for_rag()
    else:
        logger.info("RAG-prepared data already exists. Skipping preparation step.")

    # 4: Embed and insert into FAISS (if needed)
    embedder = FaissEmbedder(rag_output, index_file=faiss_index_file)
    if not os.path.exists(faiss_index_file):
        logger.info("Starting embedding and insertion into FAISS...")
        embedder.embed_and_insert()
    else:
        logger.info("FAISS index already exists. Skipping embedding and insertion step.")

    logger.info("All preprocessing steps completed.")

    # Start search and answer
    embedder.interactive_search_and_answer()
