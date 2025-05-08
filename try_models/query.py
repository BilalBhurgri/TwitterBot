import chromadb
from transformers import pipeline # AutoModelForCausalLM, AutoTokenizer
import torch
import random
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
import tweepy
# Match your DB embedding model
from sentence_transformers import SentenceTransformer 
from collections import defaultdict 

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEXT_CHUNK_SIZE = 1000
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def find_relevant_papers(collection, query_text, n_papers=3):
    """
    Find relevant papers using semantic search.
    """
    query_embedding = embedding_model.encode(query_text).tolist()

    # Account for the TEXT_CHUNK_SIZE per paper.
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_papers*TEXT_CHUNK_SIZE,  # Get extra to account for multiple chunks per paper
        include=["metadatas", "distances"]
    )

    # Group results by paper ID
    paper_scores = defaultdict(float)
    for ids, metadata, distance in zip(results["ids"], results["metadatas"], results["distances"][0]):
        paper_id = ids[0].split("_")[0] # get arxiv_id, which is before _
        # Convert distance to similarity score (cosine similarity assumed)
        paper_scores[paper_id] += 1 - distance

    # Get top N papers
    sorted_papers = sorted(paper_scores.items(), key=lambda x: x[1], reverse=True)[:n_papers]
    return [paper_id for paper_id, _ in sorted_papers]

def get_recent_papers(collection, days=60, n_papers=3):
    """Get papers published in the last N days"""
    # Query for all papers and filter by date
    # This approach works for smaller collections; for larger ones,
    # you might want to add publication date to the metadata and query by that
    results = collection.query(
        query_texts=[""], # Empty query to get all papers
        n_results=1000, # Adjust based on your collection size
        include=["metadatas"],
        where = {"chunk_index": {"$eq": 0}} # Get the abstract chunk
    )

    recent_papers = set()
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

    for metadata, doc_id in zip(results["metadatas"][0], results["ids"][0]):
        paper_metadata = json.loads(metadata["metadata"])
        pub_date = datetime.fromisoformat(paper_metadata["published"])
        if pub_date >= cutoff_date:
            # Extract the paper ID from the chunk ID
            paper_id = doc_id.split("_")[0]
            recent_papers.add(paper_id)

    return random.sample(list(recent_papers), n_papers)

def find_relevant_papers_by_abstract(collection, query_text, n_papers=3):
    """Find relevant papers by using semantic search (abstract only)"""
    query_embedding = embedding_model.encode(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_papers,
        include=["metadatas"],
        where = {"chunk_index": {"$eq": 0}} # Get the abstract chunk
    )

    papers = [doc_id.split("_")[0] for doc_id in results["ids"][0]]
    return papers

def get_paper_metadata(collection, paper_id):
    """Get metadata from the abstract chunk"""
    abstract_chunk = collection.get(ids=[f"{paper_id}_0"])["metadatas"][0]
    return json.loads(abstract_chunk["metadata"])

def generate_paper_summary(summarizer, collection, paper_id, query="novel OR breakthrough OR state-of-the-art"):
    """
    Generate summary from paper's most relevant chunks. This is after the user's query
    is passed to find_relevant_papers, so the new query for this paper is...?
    Parameters:
    summarizer - the summarizer model
    collection - 
    paper_id - arxiv_id 
    query - string for one query.
    """
    # Query the paper's own chunks for key findings
    print(f"Querying for paper {paper_id}")
    internal_results = collection.query(
        # query_texts=["novel contribution OR breakthrough OR state-of-the-art"],
        query_texts=[query],
        where={"arxiv_id": {"$eq": paper_id}},
        n_results=3
    )

    # print(f"INternal results: {internal_results}")
    
    if not internal_results["documents"]:
        return None
    
    # Combine and summarize
    context = " ".join(internal_results["documents"][0])

    print(f"Context for {paper_id}: {context}")
    return summarizer(
        context,
        max_length=200,
        min_length=50,
        do_sample=False,
        truncation=True
    )[0]['summary_text']

def generate_tweet(summary, paper_metadata, paper_id):
    """Generate a tweet based on query results"""
    title = paper_metadata["title"]
    authors = paper_metadata["authors"]
    first_author = authors[0].split()[-1]  # Last name of first author
    
    if len(authors) > 1:
        author_text = f"{first_author} et al."
    else:
        author_text = first_author
    
    # Extract the most relevant part from results
    print(f"Summary: {summary}")

    
    # Generate tweet text
    tweet = f"📑 {title}" + "\n"
    tweet += f"Key finding: {summary}" + "\n"
    tweet += f"By {author_text} #DeepLearning #AI" + "\n"
    
    # Ensure tweet is within Twitter's character limit (280)
    if len(tweet) > 280:
        # Shorten the key finding
        excess = len(tweet) - 280 + 3  # +3 for the ellipsis
        tweet = f"📑 {title}" + "\n"
        tweet = f"Key finding: {summary[:-excess]}..." + "\n"
        tweet += f"By {author_text} #DeepLearning #AI" + "\n"
        tweet += f"https://arxiv.org/abs/{paper_id}" + "\n"
    
    return tweet

def query_and_generate(args):
    summarizer = pipeline("summarization", model=args.model_type)
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "db", args.folder_name)
    client = chromadb.PersistentClient(file_path)
    collection = client.get_collection(name=args.db_name)
    items = collection.get()
    # print(len(items['ids'])) # number of entries
    # print(items.keys())

    # paper_ids = find_relevant_papers(args.query, args.num_papers)
    # "Give me a useful insight about deep learning and list the papers you referenced.", 
    #           "How do larger models learn? List the papers you referenced.", 

    # Step 1: Find relevant papers based on the query
    if args.days is not None:
        query = "novel OR breakthrough OR state-of-the-art"
        paper_ids = get_recent_papers(collection, args.days, args.num_papers)
    else:
        if args.topic is not None:
            query = args.topic
            paper_ids = find_relevant_papers_by_abstract(collection, query, args.num_papers)
        else:
            query = "Tell me sota or novel findings about vision transformers"
            paper_ids = find_relevant_papers(collection, query, args.num_papers)
    
    if not paper_ids:
        print("No papers found!")
        return None
    
    tweets = []
    for paper_id in paper_ids:
        # Step 2: Get metadata
        try:
            metadata = get_paper_metadata(collection, paper_id)
        except:
            continue

        # Step 3: Generate summary for this paper
        summary = generate_paper_summary(summarizer, collection, paper_id, query)
        if not summary:
            continue

        # Step 4: Create tweet
        tweets.append((paper_id, generate_tweet(summary, metadata, paper_id)))
    return tweets

def main():
    parser = argparse.ArgumentParser(description='Generate tweets from paper database')
    parser.add_argument('--model_type', required=False, default="facebook/bart-large-cnn", help='Specify summarizer model. Default: facebook/bart-large-cnn')
    parser.add_argument('--folder_name', required=True, help='DB folder name (within db)')
    parser.add_argument('--db_name', required=False, default='papers', help='The actual DB name within chroma.sqlite3. Default is "papers"')
    parser.add_argument('--topic', default=None, help='Optional topic to focus on')
    parser.add_argument('--num_papers', type=int, default=3, help='Number of papers to tweet about')
    parser.add_argument('--days', type=int, default=None, help='Focus on papers from last N days')
    args = parser.parse_args()

    tweets = query_and_generate(args)
    for paper_id, tweet in tweets:
        print(f"Generated Tweet for {paper_id}:")
        print(tweet)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()