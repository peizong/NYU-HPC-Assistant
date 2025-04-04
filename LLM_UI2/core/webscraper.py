import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import re
import time
import random
from concurrent.futures import ThreadPoolExecutor
from requests.adapters import HTTPAdapter
from urllib3.util import Retry
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import json
from datetime import datetime
import os.path

# Scraping Configuration
MAX_WORKERS = 5  # Number of concurrent threads
MIN_DELAY = 2  # Minimum delay between requests in seconds
MAX_DELAY = 5  # Maximum delay between requests in seconds
MIN_DELAY_BETWEEN_PAGES = 1  # Minimum delay between page scrapes
MAX_DELAY_BETWEEN_PAGES = 3  # Maximum delay between page scrapes
SELENIUM_TIMEOUT = 10  # Timeout for Selenium WebDriver in seconds
MAX_RETRIES = 3  # Maximum number of retry attempts
BACKOFF_FACTOR = 0.1  # Backoff factor for retries

# Browser Configuration
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

class WebScraper:
    def __init__(self, base_url, output_folder, url_file='scraped_urls.json'):
        self.base_url = base_url
        self.output_folder = output_folder
        self.visited_urls = set()
        self.url_file = url_file
        self.scraped_urls = self.load_scraped_urls()
        self.use_selenium = False
        self.driver = None
        self.base_url_scraped = False
        
        self.base_domain = urlparse(base_url).netloc
        
        self.headers = {
            'User-Agent': USER_AGENT
        }

        # Retry 
        self.session = requests.Session()
        retries = Retry(total=MAX_RETRIES, backoff_factor=BACKOFF_FACTOR, status_forcelist=[500, 502, 503, 504])
        self.session.mount('http://', HTTPAdapter(max_retries=retries))
        self.session.mount('https://', HTTPAdapter(max_retries=retries))

        self.scraping_complete_file = os.path.join(output_folder, 'scraping_complete.flag')

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
        time.sleep(random.uniform(MIN_DELAY, MAX_DELAY))
        
        try:
            response = self.session.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException as e:
            print(f"Error fetching {url} with requests: {str(e)}. Trying with Selenium.")
            return self.get_page_content_selenium(url)

    def get_page_content_selenium(self, url):
        if not self.driver:
            chrome_options = Options()
            chrome_options.add_argument("--headless")
            chrome_options.add_argument(f"user-agent={USER_AGENT}")
            self.driver = webdriver.Chrome(options=chrome_options)

        try:
            self.driver.get(url)
            WebDriverWait(self.driver, SELENIUM_TIMEOUT).until(EC.presence_of_element_located((By.TAG_NAME, "body")))
            return self.driver.page_source
        except Exception as e:
            print(f"Error fetching {url} with Selenium: {str(e)}")
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
        
        if url in self.scraped_urls and (url != self.base_url or self.base_url_scraped):
            print(f"Already scraped: {url}")
            return []
        
        self.visited_urls.add(url)
        print(f"Scraping: {url}")
        
        content = self.get_page_content(url)
        
        if not content:
            print(f"Failed to fetch content for {url}")
            return []

        self.save_page(url, content)
        
        if url != self.base_url:
            self.scraped_urls[url] = current_time
            self.save_scraped_urls()
        
        soup = BeautifulSoup(content, 'html.parser')
        links = soup.find_all('a', href=True)
        
        new_urls = []
        for link in links:
            href = link['href']
            new_url = urljoin(self.base_url, href)
            parsed_new_url = urlparse(new_url)
            
            # Skip URLs that are likely to be noise
            if any([
                parsed_new_url.netloc != self.base_domain,  # External links
                '#' in new_url,  # Skip ALL anchor links
                new_url in self.visited_urls,  # Already visited
                not parsed_new_url.path or parsed_new_url.path == '/'  # Root/homepage variations
            ]):
                continue
                
            new_urls.append(new_url)
        
        time.sleep(random.uniform(MIN_DELAY_BETWEEN_PAGES, MAX_DELAY_BETWEEN_PAGES))
        return new_urls

    def scrape(self):
        # Create a subdirectory for this specific base URL
        base_domain_folder = os.path.join(self.output_folder, urlparse(self.base_url).netloc)
        os.makedirs(base_domain_folder, exist_ok=True)
        
        urls_to_scrape = set([self.base_url]) if self.base_url not in self.scraped_urls else set()
        all_discovered_urls = urls_to_scrape.copy()
        
        try:
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                while urls_to_scrape:
                    current_batch = list(urls_to_scrape)
                    urls_to_scrape.clear()
                    
                    new_urls_lists = list(executor.map(self.scrape_page, current_batch))
                    
                    for sublist in new_urls_lists:
                        for url in sublist:
                            if url not in self.scraped_urls and url not in all_discovered_urls:
                                urls_to_scrape.add(url)
                                all_discovered_urls.add(url)
            
            # Mark this base URL as complete
            domain_complete_file = os.path.join(base_domain_folder, 'scraping_complete.flag')
            with open(domain_complete_file, 'w') as f:
                f.write(datetime.now().isoformat())
                
        except Exception as e:
            print(f"Scraping interrupted for {self.base_url}: {str(e)}")
            raise
        finally:
            if self.driver:
                self.driver.quit()
