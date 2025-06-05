#!/usr/bin/env python3

import json
import sys
import re

def parse_twitter_thread_file(input_file, output_file):
    """
    Parse a text file with Twitter thread metadata and convert to JSON format.
    Expected format:
    - Metadata at top (link, likes, reposts, comments, paper)
    - "--- START ---" separator
    - Thread content separated by "---"
    """
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Split content into metadata section and thread content
        if "--- START ---" in content:
            metadata_section, thread_content = content.split("--- START ---", 1)
        else:
            print("Warning: No '--- START ---' found. Treating entire file as thread content.")
            metadata_section = ""
            thread_content = content
        
        # Parse metadata
        metadata = {
            "link": "",
            "likes": 0,
            "reposts": 0,
            "comments": 0,
            "paper": "",
            "author": ""
        }
        
        # Extract metadata from the header section
        for line in metadata_section.strip().split('\n'):
            line = line.strip()
            if line.startswith('link:'):
                metadata["link"] = line.replace('link:', '').strip()
            elif line.startswith('likes:'):
                likes_str = line.replace('likes:', '').strip().replace(',', '')
                metadata["likes"] = int(likes_str) if likes_str.isdigit() else 0
            elif line.startswith('reposts:'):
                reposts_str = line.replace('reposts:', '').strip().replace(',', '')
                metadata["reposts"] = int(reposts_str) if reposts_str.isdigit() else 0
            elif line.startswith('comments:'):
                comments_str = line.replace('comments:', '').strip().replace(',', '')
                metadata["comments"] = int(comments_str) if comments_str.isdigit() else 0
            elif line.startswith('paper:'):
                metadata["paper"] = line.replace('paper:', '').strip()
        
        # Extract author from link if available
        if metadata["link"]:
            # Extract username from Twitter/X URL
            match = re.search(r'x\.com/([^/]+)/', metadata["link"])
            if match:
                metadata["author"] = match.group(1)
        
        # Split thread content by "---" separators
        tweets = []
        thread_parts = thread_content.split('---')
        
        for part in thread_parts:
            part = part.strip()
            if part:  # Skip empty parts
                tweets.append(part)
        
        # Create the final JSON structure
        thread_data = {
            "link": metadata["link"],
            "likes": metadata["likes"],
            "reposts": metadata["reposts"],
            "comments": metadata["comments"],
            "paper": metadata["paper"],
            "author": metadata["author"],
            "tweets": tweets
        }
        
        # Write to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(thread_data, f, indent=4, ensure_ascii=False)
        
        print(f"Successfully converted thread to JSON format:")
        print(f"- Author: {metadata['author']}")
        print(f"- Tweets: {len(tweets)}")
        print(f"- Likes: {metadata['likes']}")
        print(f"- Reposts: {metadata['reposts']}")
        print(f"- Comments: {metadata['comments']}")
        print(f"Output written to: {output_file}")
        
    except FileNotFoundError:
        print(f"Error: Input file '{input_file}' not found.")
    except Exception as e:
        print(f"Error: {e}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
        print("Example: python script.py paste.txt thread.json")
        print("\nExpected input format:")
        print("link: https://x.com/username/status/...")
        print("likes: 224")
        print("reposts: 46")
        print("comments: 7")
        print("paper: https://arxiv.org/abs/...")
        print("")
        print("--- START ---")
        print("")
        print("First tweet content...")
        print("")
        print("---")
        print("")
        print("Second tweet content...")
        sys.exit(1)
    
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    
    parse_twitter_thread_file(input_file, output_file)

if __name__ == "__main__":
    main()