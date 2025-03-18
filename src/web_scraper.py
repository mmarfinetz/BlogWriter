import os
import json
import time
from typing import List, Dict, Any, Optional, Union
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# Load environment variables
load_dotenv()

class FirecrawlScraper:
    """
    A class to scrape content from websites using the Firecrawl API.
    This allows collecting writing samples directly from URLs.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FirecrawlScraper with API key.

        Args:
            api_key: Firecrawl API key. If None, will try to load from
                    FIRECRAWL_API_KEY env variable.
        """
        self.api_key = api_key or os.environ.get("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("Firecrawl API key not provided. Set FIRECRAWL_API_KEY env variable or pass to constructor.")

        self.base_url = "https://api.firecrawl.dev/v1"
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def scrape_url(self, url: str) -> Dict[str, Any]:
        """
        Scrape a single URL and get its markdown content.

        Args:
            url: URL to scrape

        Returns:
            Dictionary containing the scraped content
        """
        endpoint = f"{self.base_url}/scrape"
        payload = {
            "url": url,
            "formats": ["markdown"]
        }

        response = requests.post(endpoint, json=payload, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Error scraping URL: {response.text}")

        return response.json()["data"]

    def crawl_url(self, url: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Crawl a website and get markdown content from multiple pages.

        Args:
            url: Base URL to crawl
            limit: Maximum number of pages to crawl

        Returns:
            List of dictionaries containing the scraped content
        """
        # Start the crawl
        endpoint = f"{self.base_url}/crawl"
        payload = {
            "url": url,
            "limit": limit,
            "scrapeOptions": {
                "formats": ["markdown"]
            }
        }

        response = requests.post(endpoint, json=payload, headers=self.headers)
        if response.status_code != 200:
            raise Exception(f"Error starting crawl: {response.text}")

        crawl_id = response.json()["id"]

        # Poll for results
        all_data = []
        next_url = f"{self.base_url}/crawl/{crawl_id}"
        
        print("Crawling website... This may take a few minutes.")
        progress = tqdm(total=limit, desc="Pages scraped")
        
        while True:
            response = requests.get(next_url, headers=self.headers)
            if response.status_code != 200:
                raise Exception(f"Error checking crawl status: {response.text}")

            result = response.json()
            status = result.get("status")

            if status == "completed" or "data" in result:
                if "data" in result:
                    new_data = result["data"]
                    all_data.extend(new_data)
                    progress.update(len(new_data))

                if "next" in result:
                    next_url = result["next"]
                else:
                    break
            elif status == "failed":
                raise Exception(f"Crawl failed: {result}")
            else:
                # Still processing, wait and try again
                time.sleep(5)
        
        progress.close()
        return all_data

    def save_to_files(self, data: Union[List[Dict[str, Any]], Dict[str, Any]], output_dir: str) -> List[str]:
        """
        Save scraped content to text files.

        Args:
            data: Scraped content data
            output_dir: Directory to save files

        Returns:
            List of paths to saved files
        """
        os.makedirs(output_dir, exist_ok=True)
        saved_files = []

        if isinstance(data, dict):
            # Single URL scrape result
            filename = f"scraped_{int(time.time())}.txt"
            filepath = os.path.join(output_dir, filename)
            
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(data.get("markdown", ""))
            
            saved_files.append(filepath)
        else:
            # Multiple URLs from crawl
            for i, item in enumerate(data):
                # Create a filename based on title or URL
                title = item.get("title", "")
                if title:
                    # Clean the title to create a valid filename
                    title = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
                    title = title[:50]  # Limit length
                    filename = f"{title}_{int(time.time())}_{i}.txt"
                else:
                    filename = f"scraped_{int(time.time())}_{i}.txt"
                
                filepath = os.path.join(output_dir, filename)
                
                with open(filepath, "w", encoding="utf-8") as f:
                    f.write(item.get("markdown", ""))
                
                saved_files.append(filepath)

        return saved_files

    def clean_content(self, content: str) -> str:
        """
        Clean scraped content for better training data quality.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Cleaned content
        """
        # Remove URLs
        content = content.replace("[", "").replace("]", "")
        
        # Remove HTML artifacts that might remain
        content = content.replace("&nbsp;", " ")
        content = content.replace("&amp;", "&")
        content = content.replace("&lt;", "<")
        content = content.replace("&gt;", ">")
        
        # Remove extra whitespace
        content = "\n".join(line.strip() for line in content.split("\n"))
        content = "\n".join(filter(bool, content.split("\n")))
        
        return content

    def scrape_and_save(self, url: str, output_dir: str = "./data", crawl: bool = False, limit: int = 10) -> List[str]:
        """
        Scrape content from a URL or website and save to files.
        
        Args:
            url: The URL to scrape or crawl
            output_dir: Directory to save files
            crawl: Whether to crawl the entire website
            limit: Maximum number of pages to crawl (if crawl=True)
            
        Returns:
            List of paths to saved files
        """
        try:
            if crawl:
                print(f"Crawling website: {url} (up to {limit} pages)")
                data = self.crawl_url(url, limit)
            else:
                print(f"Scraping single URL: {url}")
                data = self.scrape_url(url)
            
            saved_files = self.save_to_files(data, output_dir)
            print(f"Saved {len(saved_files)} files to {output_dir}")
            return saved_files
            
        except Exception as e:
            print(f"Error during scraping: {str(e)}")
            return []