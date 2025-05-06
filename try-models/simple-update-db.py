import chromadb
from sentence_transformers import SentenceTransformer
import arxiv
import fitz
import argparse
import os
import time
import json

# Simplified setup
parser = argparse.ArgumentParser(description='Create/update Chroma DB')
parser.add_argument('--name', required=True, help='DB name')
parser.add_argument('--input', required=True, help='File with arXiv URLs')
args = parser.parse_args()

# Initialize components
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
model = SentenceTransformer(EMBEDDING_MODEL)  # Direct model usage

client = chromadb.PersistentClient(path=f"./db/{args.name}")
collection = client.get_or_create_collection(name="papers")
TEXT_CHUNK_SIZE = 1000  # Can make this CLI arg if needed

def get_or_fetch_metadata(arxiv_id):
    """Get paper metadata with local caching"""
    cache_path = f"./cache/{arxiv_id}.json"
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    
    # Fetch from arXiv API
    paper = next(arxiv.Search(id_list=[arxiv_id]).results())
    metadata = {
        "title": paper.title,
        "authors": [str(a) for a in paper.authors],
        "abstract": paper.summary,
        "subjects": paper.categories,
        "published": paper.published.isoformat()
    }
    
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
        with fitz.open(pdf_path) as doc:
            text = " ".join([page.get_text() for page in doc])
        
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

def chunk_text(text, chunk_size=TEXT_CHUNK_SIZE, overlap=200):
    """Simple text chunker with overlap"""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

# Main processing loop
with open(args.input) as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            print(f"Processing {line.strip()}")
            process_paper(line.strip())

print("✅ Done processing papers")
