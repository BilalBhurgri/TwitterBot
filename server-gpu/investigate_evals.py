import sys
import json
import os
from pathlib import Path
import boto3
from botocore.exceptions import ClientError
from datetime import datetime
import random 
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize S3 client
s3 = boto3.client('s3', region_name='us-west-1')

def get_all_bot_objects(bucket_name, base_prefix="results-eval/Qwen/Qwen3-4B", date=None):
    """
    Get all JSON objects for bot0 through bot5 paths
    """
    all_objects = {}
    if not date:
        date = datetime.now().strftime("%Y-%m-%d")
    
    for i in range(6):  # bot0 to bot5
        bot_path = f"{base_prefix}/bot{i}/{date}"
        print(f"Checking path: {bot_path}")
        
        try:
            # Use Prefix instead of Key for listing objects in a "folder"
            response = s3.list_objects_v2(
                Bucket=bucket_name,
                Prefix=bot_path
            )
            
            bot_objects = []
            
            if 'Contents' in response:
                for obj in response['Contents']:
                    key = obj['Key']
                    # Filter for JSON files only
                    if key.endswith('.json'):
                        bot_objects.append({
                            'key': key,
                            'size': obj['Size'],
                            'last_modified': obj['LastModified'],
                            'etag': obj['ETag']
                        })
                
                print(f"Found {len(bot_objects)} JSON files in bot{i}")
                all_objects[f'bot{i}'] = bot_objects
            else:
                print(f"No objects found in bot{i}")
                all_objects[f'bot{i}'] = []
                
        except ClientError as e:
            print(f"Error accessing bot{i}: {e}")
            all_objects[f'bot{i}'] = []
    
    return all_objects

def download_json_objects(bucket_name, objects_dict, download_dir="downloads"):
    """
    Download all JSON objects to local directory
    """
    Path(download_dir).mkdir(exist_ok=True)
    
    for bot_name, objects in objects_dict.items():
        bot_dir = Path(download_dir) / bot_name
        bot_dir.mkdir(exist_ok=True)
        
        for obj in objects:
            key = obj['key']
            filename = Path(key).name
            local_path = bot_dir / filename
            
            try:
                s3.download_file(bucket_name, key, str(local_path))
                print(f"Downloaded: {key} -> {local_path}")
            except ClientError as e:
                print(f"Error downloading {key}: {e}")

def main():
    parser = argparse.ArgumentParser(description='List and download S3 objects for bot paths')
    parser.add_argument('--download_folder', default='downloads', help='Download the JSON files. Default is downloads')
    parser.add_argument('--date', required=False, help='Date path (default: yyyy-mm-dd today)')
    parser.add_argument('--prefix', default='results-eval/Qwen/Qwen3-4B', help='Specify which folder to get objects from')
    
    args = parser.parse_args()
    
    bucket_name = os.environ.get('BUCKET_NAME')
    if not bucket_name:
        print("Error: BUCKET_NAME environment variable not set")
        return
    
    # Get all objects
    all_objects = get_all_bot_objects(bucket_name, base_prefix=args.prefix, date=args.date)
    
    # Print summary
    total_files = sum(len(objects) for objects in all_objects.values())
    print(f"\nSummary:")
    print(f"Total JSON files found: {total_files}")
    for bot_name, objects in all_objects.items():
        print(f"  {bot_name}: {len(objects)} files")
    
    # Always download files
    print("\nDownloading files...")
    download_json_objects(bucket_name, all_objects, download_dir=args.download_folder)
    
    # Return the objects dictionary for further processing
    return all_objects

if __name__ == "__main__":
    objects = main()
    
    # Example: Access specific bot's objects
    # for obj in objects['bot0']:
    #     print(f"Bot0 file: {obj['key']}")