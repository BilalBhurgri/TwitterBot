import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from urllib.parse import urlparse
import requests
from io import BytesIO
import fitz
from bs4 import BeautifulSoup
import argparse

parser = argparse.ArgumentParser(
    prog='UpdateDB',
    description='Create or update your Chroma DB.',
)
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--text_chunk_size', type=int, required=True)
parser.add_argument('--input_file', type=str, required=True)

args = parser.parse_args()
NAME = args.name
TEXT_CHUNK_SIZE = args.text_chunk_size
INPUT_FILE = args.input_file


# Set up ChromaDB and init embeddings
client = chromadb.PersistentClient(path="./db/" + NAME)
collection = client.get_or_create_collection(name="papers")
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",  # Model name as string
    model_kwargs={"device": "cpu"},  # Optional device settings
    encode_kwargs={"normalize_embeddings": False}  # Optional encoding flags
)


def index_papers(file_name):
    """
    Index a list of research papers into ChromaDB using Langchain embeddings.
    It chunks text and embeds each chunk into a vector.
    Args:
        file_name (f): The name of a file with a direct list of links to direct PDFs. 
        Preferably arxiv ones!
    """

    # Index papers
    with open(file_name) as f:
        # Convert full paper + abstract + metadata into embeddings
        lines = f.readlines()
        for line in lines:
            if(line.startswith("#")):
                continue
            print(f"Processing {line}")
            text = fetch_and_parse_pdf(line)
            if not text.strip():
                print(f"No text found for {line}")
                continue

            title, authors, abstract, subjects = fetch_arxiv_metadata(line)

            chunks = chunk_text(text, chunk_size=1000, overlap=200)
            vectors = embeddings.embed_documents(chunks)

            ids = [f"{hash(title)}_{i}" for i in range(len(chunks))]
            metadatas = [{"title": title, "authors": authors, "subjects": subjects} for _ in chunks]

            # Store paper in ChromaDB collection, with repeat metadata for each text chunk
            collection.add(
                documents=chunks,  # For human readable results
                metadatas=metadatas,
                ids=ids, # A unique identifier for each paper
                embeddings=vectors  # For search
            )

def chunk_text(text,overlap=200):
    """
    Split the text into overlapping chunks for vector embedding.
    """
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + TEXT_CHUNK_SIZE)
        chunk = text[start:end]
        chunks.append(chunk)
        start += TEXT_CHUNK_SIZE - overlap
    return chunks

def fetch_and_parse_pdf(url):
    """
    This fetches the paper.
    Returns:
        str: The full text of the paper.
    """
    response = requests.get(url)
    if not response.ok:
        print(f"❌ Failed to fetch {url}")
        return ""

    with BytesIO(response.content) as f:
        doc = fitz.open(stream=f.read(), filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    
def fetch_arxiv_metadata(url):
    """
    Fetch metadata from an arXiv paper given its URL.
    Args:
        url (str): The URL of the arXiv paper, either in PDF form or abstract page.
    Returns:
        list: A list containing the title, authors, abstract, and subjects of the paper.
              Returns an empty string if the abstract page cannot be fetched.
    """
    if "pdf" in url:
        arxiv_id = url.split("/pdf/")[-1].replace(".pdf", "")
        print(f"arxiv_id is {arxiv_id}")
        abs_url = f"https://arxiv.org/abs/{arxiv_id}"
    else:
        abs_url = url

    print(f"Fetching abstract page from {abs_url}")
    response = requests.get(abs_url)
    if not response.ok:
        print(f"❌ Failed to fetch abstract page: {abs_url}")
        return None
    
    soup = BeautifulSoup(response.text, "html.parser")
    title = soup.find("h1", class_="title mathjax")
    authors = soup.find("div", class_="authors")
    abs_block = soup.find("blockquote", class_="abstract mathjax")
    subjects = fetch_arxiv_subjects(soup)

    return [title.text, authors.text, abs_block.text, subjects]

def fetch_arxiv_subjects(soup):
    table_rows = soup.select("div.metatable tr")
    for row in table_rows:
        header_cell = row.find("td", class_="tablecell label")
        if header_cell and header_cell.text.strip() == "Subjects:":
            value_cell = row.find_all("td")[-1]
            print(f"Found the subject/s: {value_cell.text.strip()}")
            return value_cell.text.strip()
    return ""


def main():
    index_papers(INPUT_FILE)

main()