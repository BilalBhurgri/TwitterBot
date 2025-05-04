import os
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from sentence_transformers import SentenceTransformer
import chromadb
from langchain.llms import OpenAI

# Step 1: Set up ChromaDB client and collection
client = chromadb.Client()

# Create or connect to a collection where documents will be stored
collection = client.create_collection(name="research_papers")

# Step 2: Define a function to index research papers
def index_research_papers(papers):
    """
    Index a list of research papers into ChromaDB using Langchain embeddings.
    Args:
        papers (list): List of tuples where each tuple is (title, abstract)
    """
    # Initialize Sentence-Transformers model for embedding
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = SentenceTransformerEmbeddings(model)

    # Index papers
    for title, abstract in papers:
        # Convert text (abstract) to embeddings
        vector = embeddings.embed([abstract])[0]

        # Store paper in ChromaDB collection
        collection.add(
            documents=[abstract],  # Text content
            metadatas=[{"title": title}],  # Metadata for retrieval
            ids=[str(hash(title))],  # A unique identifier for each paper
            embeddings=[vector]  # The embeddings of the abstract
        )

# Sample research papers (title, abstract)
papers = [
    ("Paper 1", "This is the abstract of paper 1."),
    ("Paper 2", "This is the abstract of paper 2."),
    # Add more papers as needed
]

# Index papers
index_research_papers(papers)

# Step 3: Define a function to query ChromaDB and retrieve relevant papers
def query_research_papers(query, top_k=3):
    """
    Query the indexed research papers in ChromaDB.
    Args:
        query (str): The query text to search for relevant papers.
        top_k (int): Number of top documents to retrieve.
    """
    # Get embeddings for the query
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = SentenceTransformerEmbeddings(model)
    query_vector = embeddings.embed([query])[0]

    # Retrieve top_k documents from ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k
    )
    
    return results['documents'], results['metadatas']

# Step 4: Use Langchain to process and summarize results
def generate_summary_from_results(results):
    """
    Use Langchain and OpenAI LLM to summarize the relevant papers retrieved from ChromaDB.
    Args:
        results (list): List of documents from ChromaDB to summarize.
    """
    # Create Langchain PromptTemplate for summarization
    prompt_template = "Summarize the following research papers:\n\n{documents}"
    prompt = PromptTemplate(input_variables=["documents"], template=prompt_template)

    # Create Langchain LLMChain using OpenAI's GPT model
    llm = OpenAI(temperature=0.7)  # OpenAI model can be swapped with other LLMs
    chain = LLMChain(llm=llm, prompt=prompt)

    # Combine all papers' content into one string for summarization
    documents_text = "\n\n".join(results)
    
    # Generate summary
    summary = chain.run(documents=documents_text)
    return summary

# Step 5: Save summaries into a folder
def save_summary_to_folder(summary, folder_path, query):
    """
    Save the generated summary to a text file in the specified folder.
    Args:
        summary (str): The summary to save.
        folder_path (str): The folder where to save the summary.
        query (str): The query used to retrieve papers (used for filename).
    """
    # Ensure the folder exists
    os.makedirs(folder_path, exist_ok=True)

    # Create a filename based on the query
    filename = f"summary_{query.replace(' ', '_')}.txt"
    file_path = os.path.join(folder_path, filename)

    # Write the summary to a file
    with open(file_path, "w") as file:
        file.write(summary)

    print(f"Summary saved to {file_path}")

# Step 6: Query papers and generate summary
query = "What are the latest advancements in AI?"
top_k_results, metadata = query_research_papers(query)

# Extract document texts
documents = [doc for doc in top_k_results]

# Generate summary of top papers
summary = generate_summary_from_results(documents)

# Define the folder path where summaries will be saved
folder_path = "summaries"

# Save the summary to the folder
save_summary_to_folder(summary, folder_path, query)
