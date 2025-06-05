#!/usr/bin/env python3
"""
ArXiv Paper Downloader for Twitter Thread Dataset

This script extracts arXiv paper URLs from your JSON dataset and downloads them.
It handles various URL formats and provides detailed logging.
"""

import json
import re
import requests
import os
from pathlib import Path
import time
from urllib.parse import urlparse
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('arxiv_downloads.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ArXivDownloader:
    def __init__(self, json_file_path, download_dir="arxiv_papers"):
        """
        Initialize the ArXiv downloader
        
        Args:
            json_file_path (str): Path to your JSON file
            download_dir (str): Directory to save papers
        """
        self.json_file_path = json_file_path
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Pattern to extract arXiv ID from various URL formats
        self.arxiv_patterns = [
            r'arxiv\.org/abs/(\d+\.\d+)',           # https://arxiv.org/abs/2505.17968
            r'arxiv\.org/pdf/(\d+\.\d+)',           # https://arxiv.org/pdf/2505.17968
            r'arxiv\.org/abs/(\d+\.\d+v\d+)',       # https://arxiv.org/abs/2505.17968v1
            r'arxiv\.org/pdf/(\d+\.\d+v\d+)',       # https://arxiv.org/pdf/2505.17968v1
        ]
    
    def load_threads(self):
        """Load threads from JSON file - handles messy JSON"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Clean up common JSON issues
            content = content.strip()
            if content.endswith('}   '):
                content = content[:-3]
            
            # Try to parse
            data = json.loads(content)
            threads = data.get('threads', [])
            logger.info(f"Successfully loaded {len(threads)} threads")
            return threads
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error at line {e.lineno}: {e.msg}")
            logger.error("Trying to extract URLs manually...")
            return self.extract_urls_manually()
        except Exception as e:
            logger.error(f"Error loading file: {e}")
            return []

    def extract_urls_manually(self):
        """Fallback: manually extract URLs if JSON parsing fails"""
        try:
            with open(self.json_file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            threads = []
            
            # Find all paper URLs
            paper_matches = re.findall(r'"paper":\s*"([^"]+)"', content)
            author_matches = re.findall(r'"author":\s*"([^"]+)"', content)
            link_matches = re.findall(r'"link":\s*"([^"]+)"', content)
            
            # Combine them
            for i, (paper, author, link) in enumerate(zip(paper_matches, author_matches, link_matches)):
                threads.append({
                    'paper': paper,
                    'author': author,
                    'link': link
                })
            
            logger.info(f"Manually extracted {len(threads)} threads")
            return threads
            
        except Exception as e:
            logger.error(f"Manual extraction failed: {e}")
            return []
    
    def extract_arxiv_id(self, url):
        """
        Extract arXiv ID from various URL formats
        
        Args:
            url (str): Paper URL
            
        Returns:
            str or None: arXiv ID if found
        """
        if not url or not isinstance(url, str):
            return None
            
        for pattern in self.arxiv_patterns:
            match = re.search(pattern, url)
            if match:
                arxiv_id = match.group(1)
                # Remove version number if present for consistent naming
                arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
                return arxiv_id
        
        return None
    
    def download_paper(self, arxiv_id, author, thread_link):
        """
        Download a paper from arXiv
        
        Args:
            arxiv_id (str): arXiv paper ID
            author (str): Twitter author handle
            thread_link (str): Original thread URL
            
        Returns:
            bool: Success status
        """
        # Create filename: arxiv_id_author.pdf
        filename = f"{arxiv_id}_{author}.pdf"
        filepath = self.download_dir / filename
        
        # Skip if already downloaded
        if filepath.exists():
            logger.info(f"Paper already exists: {filename}")
            return True
        
        # Construct arXiv PDF URL
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        
        try:
            logger.info(f"Downloading {arxiv_id} by @{author}...")
            
            # Download with headers to appear like a browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            response = requests.get(pdf_url, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Write PDF file
            with open(filepath, 'wb') as f:
                f.write(response.content)
            
            logger.info(f"✅ Downloaded: {filename} ({len(response.content)} bytes)")
            
            # Create metadata file
            metadata = {
                'arxiv_id': arxiv_id,
                'author': author,
                'thread_link': thread_link,
                'pdf_url': pdf_url,
                'filename': filename
            }
            
            metadata_path = self.download_dir / f"{arxiv_id}_{author}_metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return True
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Failed to download {arxiv_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error downloading {arxiv_id}: {e}")
            return False
    
    def process_all_threads(self, delay=1.0):
        """
        Process all threads and download papers
        
        Args:
            delay (float): Delay between downloads (seconds)
        """
        threads = self.load_threads()
        
        if not threads:
            logger.error("No threads found in JSON file")
            return
        
        logger.info(f"Found {len(threads)} threads to process")
        
        successful_downloads = 0
        failed_downloads = 0
        skipped_count = 0
        
        for i, thread in enumerate(threads, 1):
            author = thread.get('author', 'unknown')
            paper_url = thread.get('paper', '')
            thread_link = thread.get('link', '')
            
            logger.info(f"\n[{i}/{len(threads)}] Processing thread by @{author}")
            
            # Extract arXiv ID
            arxiv_id = self.extract_arxiv_id(paper_url)
            
            if not arxiv_id:
                logger.warning(f"⚠️  No arXiv ID found in URL: {paper_url}")
                skipped_count += 1
                continue
            
            # Download paper
            if self.download_paper(arxiv_id, author, thread_link):
                successful_downloads += 1
            else:
                failed_downloads += 1
            
            # Respectful delay
            if i < len(threads):
                time.sleep(delay)
        
        # Summary
        logger.info(f"\n📊 DOWNLOAD SUMMARY:")
        logger.info(f"✅ Successful: {successful_downloads}")
        logger.info(f"❌ Failed: {failed_downloads}")
        logger.info(f"⚠️  Skipped: {skipped_count}")
        logger.info(f"📁 Papers saved to: {self.download_dir.absolute()}")
    
    def create_bibliography(self):
        """Create a bibliography file from downloaded papers"""
        threads = self.load_threads()
        bib_entries = []
        
        for thread in threads:
            author = thread.get('author', 'unknown')
            paper_url = thread.get('paper', '')
            arxiv_id = self.extract_arxiv_id(paper_url)
            
            if arxiv_id:
                # Simple bibliography entry
                entry = f"arXiv:{arxiv_id} (Twitter thread by @{author}): {paper_url}"
                bib_entries.append(entry)
        
        # Write bibliography
        bib_path = self.download_dir / "bibliography.txt"
        with open(bib_path, 'w') as f:
            f.write("ArXiv Papers from Twitter Thread Dataset\n")
            f.write("=" * 50 + "\n\n")
            for entry in bib_entries:
                f.write(f"• {entry}\n")
        
        logger.info(f"📚 Bibliography created: {bib_path}")

def main():
    """Main function"""
    # Configuration
    JSON_FILE = "all_threads.json"  # Your JSON file name
    DOWNLOAD_DIR = "arxiv_papers"
    DELAY = 2.0  # Seconds between downloads (be respectful to arXiv)
    
    # Create downloader
    downloader = ArXivDownloader(JSON_FILE, DOWNLOAD_DIR)
    
    # Download all papers
    downloader.process_all_threads(delay=DELAY)
    
    # Create bibliography
    downloader.create_bibliography()
    
    print(f"\n🎉 Done! Check the '{DOWNLOAD_DIR}' folder for your papers.")

if __name__ == "__main__":
    main()