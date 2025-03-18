#!/usr/bin/env python3

import argparse
import os
import sys
from dotenv import load_dotenv
from src.web_scraper import FirecrawlScraper

def parse_args():
    parser = argparse.ArgumentParser(description="Collect writing samples from websites using Firecrawl API")
    parser.add_argument(
        "--url", 
        type=str, 
        required=True,
        help="URL to scrape or crawl"
    )
    parser.add_argument(
        "--crawl", 
        action="store_true", 
        help="Crawl the entire website instead of just scraping a single URL"
    )
    parser.add_argument(
        "--limit", 
        type=int, 
        default=10, 
        help="Maximum number of pages to crawl (only applicable with --crawl)"
    )
    parser.add_argument(
        "--api_key", 
        type=str, 
        default=None, 
        help="Firecrawl API key (will use FIRECRAWL_API_KEY env variable if not provided)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data", 
        help="Directory to save scraped content"
    )
    
    return parser.parse_args()

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    args = parse_args()
    
    # Print collection configuration
    print(f"Sample collection configuration:")
    print(f"URL: {args.url}")
    print(f"Mode: {'Crawl' if args.crawl else 'Single URL'}")
    if args.crawl:
        print(f"Page limit: {args.limit}")
    print(f"Output directory: {args.output_dir}")
    
    # Check if API key is available
    api_key = args.api_key or os.environ.get("FIRECRAWL_API_KEY")
    if not api_key:
        print("Error: Firecrawl API key not provided.")
        print("Either set the FIRECRAWL_API_KEY environment variable, create a .env file, or use the --api_key argument.")
        sys.exit(1)
    
    # Initialize scraper
    try:
        scraper = FirecrawlScraper(api_key=api_key)
    except ValueError as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
    
    # Scrape and save content
    try:
        saved_files = scraper.scrape_and_save(
            url=args.url,
            output_dir=args.output_dir,
            crawl=args.crawl,
            limit=args.limit
        )
        
        if saved_files:
            print("\nCollection complete!")
            print(f"Collected {len(saved_files)} writing samples.")
            print(f"Files saved to: {args.output_dir}")
            print("\nNext steps:")
            print(f"  - Review the collected samples in {args.output_dir}")
            print(f"  - Train your model: python train.py --data_dir {args.output_dir}")
        else:
            print("\nNo samples were collected. Please check the URL and try again.")
    except Exception as e:
        print(f"Error during scraping: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()