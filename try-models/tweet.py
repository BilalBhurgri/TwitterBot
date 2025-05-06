import chromadb
from sentence_transformers import SentenceTransformer
import random
import argparse
import json
import os
from datetime import datetime, timedelta, timezone
import tweepy  # You'll need to install this

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate tweets from paper database')
parser.add_argument('--name', required=True, help='DB name')
parser.add_argument('--topic', default=None, help='Optional topic to focus on')
parser.add_argument('--days', type=int, default=30, help='Focus on papers from last N days')
parser.add_argument('--post', action='store_true', help='Actually post to Twitter')
args = parser.parse_args()

# Initialize components
client = chromadb.PersistentClient(path=f"./db/{args.name}")
collection = client.get_or_create_collection(name="papers")
model = SentenceTransformer("all-MiniLM-L6-v2")

# Twitter API credentials - store these securely in practice
# Consider using environment variables or a secure config file
twitter_auth = {
    "consumer_key": os.environ.get("TWITTER_API_KEY"),
    "consumer_secret": os.environ.get("TWITTER_API_SECRET"),
    "access_token": os.environ.get("TWITTER_ACCESS_TOKEN"),
    "access_token_secret": os.environ.get("TWITTER_ACCESS_SECRET")
}

def setup_twitter_api():
    """Set up and return Twitter API client"""
    auth = tweepy.OAuth1UserHandler(
        twitter_auth["consumer_key"], twitter_auth["consumer_secret"],
        twitter_auth["access_token"], twitter_auth["access_token_secret"]
    )
    return tweepy.API(auth)

def get_recent_papers(days=30):
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
            print(paper_id)
            recent_papers.add(paper_id)

    return list(recent_papers)

def query_papers(query_text, n_results=3):
    """Query the database for relevant paper chunks"""
    query_embedding = model.encode(query_text)
    
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=n_results,
        include=["documents", "metadatas"],
        where = {"chunk_index": {"$eq": 0}} # Get the abstract chunk
    )
    
    return results

def extract_interesting_finding(paper_id):
    """Extract the most interesting finding from a paper"""
    # First get the abstract
    abstract_results = collection.query(
        query_texts=[""],  # Empty query
        where={"id": {"$eq": f"{paper_id}_0"}},  # Get the abstract chunk
        include=["documents", "metadatas"]
    )
    
    if not abstract_results["documents"]:
        return None, None
    
    # Get paper metadata
    paper_metadata = json.loads(abstract_results["metadatas"][0]["metadata"])
    
    # Now query with a prompt designed to find interesting parts
    interesting_prompts = [
        "novel findings", 
        "breakthrough results",
        "state-of-the-art performance",
        "innovative approach",
        "surprising conclusion"
    ]
    
    # Choose a random prompt for variety
    prompt = random.choice(interesting_prompts)
    
    # Query the paper chunks
    results = collection.query(
        query_texts=[prompt],
        where={"$and": [{"id": {"$contains": paper_id}}]},
        n_results=3,
        include=["documents", "metadatas"]
    )
    
    return results, paper_metadata

def generate_tweet(results, paper_metadata):
    """Generate a tweet based on query results"""
    title = paper_metadata["title"]
    authors = paper_metadata["authors"]
    first_author = authors[0].split()[-1]  # Last name of first author
    
    if len(authors) > 1:
        author_text = f"{first_author} et al."
    else:
        author_text = first_author
    
    # Extract the most relevant part from results
    key_finding = results["documents"][0][:200]  # First 200 chars of top result
    
    # Generate tweet text
    tweet = f"📑 {title}\n\n"
    tweet += f"Key finding: {key_finding.strip()}...\n\n"
    tweet += f"By {author_text} #DeepLearning #MachineLearning\n"
    
    # Add arXiv link
    arxiv_id = results["metadatas"][0]["id"].split("_")[0]
    tweet += f"https://arxiv.org/abs/{arxiv_id}"
    
    # Ensure tweet is within Twitter's character limit (280)
    if len(tweet) > 280:
        # Shorten the key finding
        excess = len(tweet) - 280 + 3  # +3 for the ellipsis
        tweet = f"📑 {title}\n\n"
        tweet += f"Key finding: {key_finding[:200-excess].strip()}...\n\n"
        tweet += f"By {author_text} #DeepLearning #MachineLearning\n"
        tweet += f"https://arxiv.org/abs/{arxiv_id}"
    
    return tweet

def main():
    # Choose a random recent paper if no specific topic
    if args.topic is None:
        recent_papers = get_recent_papers(days=args.days)
        if not recent_papers:
            print("No recent papers found!")
            return
        paper_id = random.choice(recent_papers)
        results, paper_metadata = extract_interesting_finding(paper_id)
    else:
        # Query based on topic
        topic_results = query_papers(args.topic, n_results=3)
        if not topic_results["documents"]:
            print(f"No relevant papers found for topic: {args.topic}")
            return
        
        # Extract paper ID from a random result
        print(topic_results["ids"][0])
        paper_id = random.choice(topic_results["ids"][0]).split("_")[0]
        results, paper_metadata = extract_interesting_finding(paper_id)
    
    if not results or not paper_metadata:
        print(f"Could not extract information from paper {paper_id}")
        return
    
    # Generate tweet
    # tweet = generate_tweet(results, paper_metadata)
    tweet = "Test Tweet"
    print("\n--- Generated Tweet ---")
    print(tweet)
    print("--- End Tweet ---\n")
    print(f"Character count: {len(tweet)}/280")
    
    # Post to Twitter if requested
    if args.post:
        try:
            api = setup_twitter_api()
            api.update_status(tweet)
            print("✅ Tweet posted successfully!")
        except Exception as e:
            print(f"❌ Failed to post tweet: {str(e)}")

if __name__ == "__main__":
    main()