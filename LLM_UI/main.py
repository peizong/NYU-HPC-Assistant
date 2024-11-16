# ***************************************************************************************

# IF YOU ARE HAVING TROUBLE WITH THE IMPORTS BELOW, CHANGE YOUR PYTHON INTERPRETER VERSION
# USE THIS COMMAND IN CMD TO SET YOUR KEY: setx OPENAI_API_KEY PUT_YOUR_KEY_HERE
# DO NOT INTERRUPT THE CODE WHILE IT IS IS INITIALIZING


# If you do not want to generate your own resources, set variable to True to download from S3
# USE_PREGENERATED_RESOURCES = False  # Set to True to download from S3
# ***************************************************************************************


import os
import logging
import os.path
import subprocess
import requests
from urllib.parse import urlparse
from pathlib import Path

from core.webscraper import WebScraper
from core.datacleaner import DataCleaner
from core.ragpreparator import RAGPreparator
from core.faisembedder import FaissEmbedder


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


#IMPORTANT: Set to True to download from S3

USE_PREGENERATED_RESOURCES = True  # Set to True to download from S3
S3_RESOURCES_URL = "https://nyu-hpc-llm.s3.us-east-1.amazonaws.com/" 

# Global Configuration Variables
BASE_URL = "https://sites.google.com/nyu.edu/nyu-hpc/"
CHUNK_SIZE = 1000  # Size of text chunks for RAG preparation

# Resource Directory Structure
RESOURCES_DIR_NAME = "resources"
SCRAPED_DATA_DIR_NAME = "scraped_data_nyu_hpc"
CLEANED_DATA_FILENAME = "cleaned_data_nyu_hpc.csv"
RAG_DATA_FILENAME = "rag_prepared_data_nyu_hpc.csv"
FAISS_INDEX_FILENAME = "faiss_index.pkl"
URL_FILENAME = "nyu_hpc_scraped_urls.json"
SCRAPING_COMPLETE_FLAG = "scraping_complete.flag"
EMBEDDING_CHECKPOINT_FILENAME = "embedding_checkpoint.json"

def download_resources(resources_dir: str, s3_base_url: str):
    """Download all resources from S3 by parsing the bucket XML listing"""
    logger.info("Downloading all resources from S3...")
    
    try:
        # Get bucket listing
        response = requests.get(s3_base_url)
        response.raise_for_status()
        
        # Create resources directory
        os.makedirs(resources_dir, exist_ok=True)
        
        # Find all Key elements in the XML
        from xml.etree import ElementTree
        root = ElementTree.fromstring(response.content)
        
        # Define the namespace
        namespace = {'s3': 'http://s3.amazonaws.com/doc/2006-03-01/'}
        
        # Download each file
        for content in root.findall('.//s3:Contents', namespace):
            key = content.find('s3:Key', namespace).text
            
            # Skip if not in resources folder
            if not key.startswith('resources/'):
                continue
                
            # Get just the filename
            filename = os.path.basename(key)
            local_path = os.path.join(resources_dir, filename)
            
            # Download file
            try:
                file_url = s3_base_url + key
                logger.info(f"Downloading {filename}...")
                
                file_response = requests.get(file_url, stream=True)
                file_response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in file_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                logger.info(f"Successfully downloaded {filename}")
                
            except requests.exceptions.RequestException as e:
                logger.warning(f"Failed to download {filename}: {str(e)}")
                continue
                
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to access S3 bucket: {str(e)}")
        raise
    except ElementTree.ParseError as e:
        logger.error(f"Failed to parse S3 bucket listing: {str(e)}")
        raise
        
    logger.info("Resource download completed")

def main():
    print("Initializing, please wait...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    resources_dir = os.path.join(script_dir, RESOURCES_DIR_NAME)

    if USE_PREGENERATED_RESOURCES:
        # Check if resources already exist
        required_files = [CLEANED_DATA_FILENAME, RAG_DATA_FILENAME, FAISS_INDEX_FILENAME]
        resources_exist = all(
            os.path.exists(os.path.join(resources_dir, file))
            for file in required_files
        )

        if not resources_exist:
            # Download only if resources are missing
            os.makedirs(resources_dir, exist_ok=True)
            download_resources(resources_dir, S3_RESOURCES_URL)
            logger.info("Successfully downloaded pre-generated resources.")
        else:
            logger.info("Pre-generated resources already exist. Skipping download.")
    else:
        # Original resource generation logic
        os.makedirs(resources_dir, exist_ok=True)
        
        # Define all paths relative to the resources directory
        output_folder = os.path.join(resources_dir, SCRAPED_DATA_DIR_NAME)
        cleaned_output = os.path.join(resources_dir, CLEANED_DATA_FILENAME)
        rag_output = os.path.join(resources_dir, RAG_DATA_FILENAME)
        faiss_index_file = os.path.join(resources_dir, FAISS_INDEX_FILENAME)
        url_file = os.path.join(resources_dir, URL_FILENAME)
        scraping_complete_file = os.path.join(resources_dir, SCRAPING_COMPLETE_FLAG)
        checkpoint_file = os.path.join(resources_dir, EMBEDDING_CHECKPOINT_FILENAME)

        # Modified check for complete scraping
        is_scraping_incomplete = (
            not os.path.exists(scraping_complete_file) or
            not os.path.exists(output_folder) or 
            not any(os.scandir(output_folder))
        )
        
        if is_scraping_incomplete:
            logger.info("Starting or resuming web scraping...")
            scraper = WebScraper(BASE_URL, output_folder, url_file=url_file)
            scraper.scrape()
        else:
            logger.info("Scraping was previously completed. Skipping scraping step.")

        # 2: Clean data (if needed)
        if not os.path.exists(cleaned_output) or not os.path.exists(f"{cleaned_output}.complete"):
            logger.info("Starting data cleaning...")
            cleaner = DataCleaner(output_folder, cleaned_output)
            cleaner.clean_data()
        else:
            logger.info("Cleaned data already exists and is complete. Skipping cleaning step.")

        # 3: Prepare for RAG (if needed)
        if not os.path.exists(rag_output) or not os.path.exists(f"{rag_output}.complete"):
            logger.info("Starting RAG preparation...")
            preparator = RAGPreparator(cleaned_output, rag_output, chunk_size=CHUNK_SIZE)
            preparator.prepare_for_rag()
        else:
            logger.info("RAG-prepared data already exists and is complete. Skipping preparation step.")

        # 4: Embed and insert into FAISS
        is_embedding_incomplete = (
            os.path.exists(checkpoint_file) or  
            not os.path.exists(faiss_index_file)        
        )
        
        if is_embedding_incomplete:
            logger.info("Starting or resuming embedding and insertion into FAISS...")
            embedder = FaissEmbedder(rag_output, index_file=faiss_index_file, checkpoint_file=checkpoint_file)
            embedder.embed_and_insert()
        else:
            logger.info("FAISS index already exists and is complete. Skipping embedding and insertion step.")

        logger.info("All preprocessing steps completed.")

    # Run the Streamlit app
    streamlit_path = os.path.join(script_dir, "streamlit_app.py")
    subprocess.run(["streamlit", "run", streamlit_path])

if __name__ == "__main__":
    main()
