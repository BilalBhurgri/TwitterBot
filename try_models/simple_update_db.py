import chromadb
from sentence_transformers import SentenceTransformer
import arxiv
import fitz
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import json
# import data_processing.parse_paper as parse_paper
# import data_processing.parse_paper_remove_math as parse_paper_remove_math
from data_processing import parse_paper
from data_processing import parse_paper_remove_math
# import parse_paper
# import parse_paper_remove_math
from try_models import get_paper_xml
import boto3
from dotenv import load_dotenv

# Simplified setup moved to main
# parser = argparse.ArgumentParser(description='Create/update Chroma DB')
# parser.add_argument('--name', required=True, help='DB name')
# parser.add_argument('--input', required=True, help='File with arXiv URLs')
# parser.add_argument('--embedding_model', required=True, help='Embedding model')
# parser.add_argument('--text_chunk_size', default=1000, help='Text chunk size')
# args = parser.parse_args()

# Initialize components
# EMBEDDING_MODEL = args.embedding_model
# model = SentenceTransformer(EMBEDDING_MODEL)  # Direct model usage

# client = chromadb.PersistentClient(path=f"./db/{args.name}")
# collection = client.get_or_create_collection(name="papers")
# TEXT_CHUNK_SIZE = 1000  # Can make this CLI arg if needed

# AWS S3
# s3 = boto3.client('s3', region_name='us-west-1')
# Global placeholders
model = None
collection = None
s3 = None
TEXT_CHUNK_SIZE = 1000


def get_or_fetch_metadata(arxiv_id):
    """Get paper metadata with local caching"""
    cache_path = f"./cache/{arxiv_id}.json"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    
    # Fetch from arXiv API
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    paper = next(client.results(search))

    metadata = {
        "title": paper.title,
        "authors": [str(a) for a in paper.authors],
        "abstract": paper.summary,
        "subjects": paper.categories,
        "published": paper.published.isoformat()
    }
    # paper = next(arxiv.Search(id_list=[arxiv_id]).results())
    # metadata = {
    #     "title": paper.title,
    #     "authors": [str(a) for a in paper.authors],
    #     "abstract": paper.summary,
    #     "subjects": paper.categories,
    #     "published": paper.published.isoformat()
    # }
    
    with open(cache_path, "w") as f:
        json.dump(metadata, f)
    
    return metadata

def process_paper(url):
    """Full processing pipeline for a single paper"""
    try:
        # Extract arXiv ID from URL
        arxiv_id = url.split("/pdf/")[-1].replace(".pdf", "").split("v")[0]
        pdf_path = f"./pdfs/{arxiv_id}.pdf"
        
        # Get metadata (cached or fresh)
        metadata = get_or_fetch_metadata(arxiv_id)
        
        # Download PDF if needed
        if not os.path.exists(pdf_path):
            print(f"Fetching PDF for {arxiv_id} from Arxiv...")
            paper = next(arxiv.Search(id_list=[arxiv_id]).results())
            paper.download_pdf(dirpath="./pdfs", filename=f"{arxiv_id}.pdf")
            time.sleep(3)  # Rate limit

        existing = collection.get(ids=[f"{arxiv_id}_0"])

        if existing and existing["ids"]:
            print(f"{arxiv_id} already in the collection. skipping!")
            return
        
        # Process text
        text = load_paper(pdf_path)
        
        # Create chunks with abstract as first chunk
        chunks = [metadata["abstract"]] + chunk_text(text)
        
        # Generate embeddings
        embeddings = model.encode(chunks)

        metadata_str = json.dumps(metadata, ensure_ascii=False)
        
        # Prepare metadata for each chunk
        chunk_metadatas = [
            {
            "arxiv_id": f"{arxiv_id}",
            "metadata": metadata_str,
            "chunk_type": "abstract" if i == 0 else "body",
            "chunk_index": i
        } for i in range(len(chunks))]
        
        # Store in Chroma
        collection.upsert(
            documents=chunks,
            embeddings=embeddings.tolist(),
            ids=[f"{arxiv_id}_{i}" for i in range(len(chunks))],
            metadatas=chunk_metadatas
        )
        
    except Exception as e:
        print(f"⚠️ Failed to process {arxiv_id}: {str(e)}")
        return None
    
def load_paper(paper_path):
    """
    Creates a temporary XML file and extracts text without math in it. 
    Saves the extracted text to a .txt file in the same directory as the PDF.
    """
    # First convert PDF to XML using GROBID
    print("GROBID ISOLATION TECHNIQUE")
    xml_content = get_paper_xml.process_pdf(paper_path)
    print("GROBID ISOLATION TECHNIQUE 2")
    if isinstance(xml_content, dict) and "error" in xml_content:
        print(f"Error processing PDF: {xml_content['error']}")
        return None
        
    # Save the XML content to a temporary file
    xml_path = paper_path.replace('.pdf', '.xml')
    try:
        with open(xml_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        # Now extract text from the XML
        text = parse_paper_remove_math.extract_text_from_xml(xml_path)
        
        # Save the extracted text to a .txt file
        txt_path = paper_path.replace('.pdf', '.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(text)
            
        try:
            s3.upload_file(txt_path, os.getenv("BUCKET_NAME"), txt_path)
            print(f"Uploaded {local_file_path} to s3://{bucket_name}/{txt_path}")
        except Exception as e:
            print(f"Upload failed: {e}")
        
        # Delete the temporary XML file
        os.remove(xml_path)
        
        return text
    except Exception as e:
        print(f"Error processing XML: {str(e)}")
        # Clean up XML file if it exists
        if os.path.exists(xml_path):
            os.remove(xml_path)
        return None

def chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, overlap=200):
    """Simple text chunker with overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def main():
    # Main processing loop
    with open(args.input) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                print(f"Processing {line.strip()}")
                process_paper(line.strip())

    print("✅ Done processing papers")

if __name__ == "__main__":
    # Simplified setup
    parser = argparse.ArgumentParser(description='Create/update Chroma DB')
    parser.add_argument('--name', required=True, help='DB name')
    parser.add_argument('--input', required=True, help='File with arXiv URLs')
    parser.add_argument('--embedding_model', required=True, help='Embedding model')
    parser.add_argument('--text_chunk_size', default=1000, help='Text chunk size')
    args = parser.parse_args()
    EMBEDDING_MODEL = args.embedding_model
    model = SentenceTransformer(EMBEDDING_MODEL)  # Direct model usage

    client = chromadb.PersistentClient(path=f"./db/{args.name}")
    collection = client.get_or_create_collection(name="papers")
    TEXT_CHUNK_SIZE = 1000
    s3 = boto3.client('s3', region_name='us-west-1')
    main()