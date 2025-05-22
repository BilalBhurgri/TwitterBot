"""Fast eval implementation using ollama"""
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
from try_models.bot_persona import BotPersona
from transformers import AutoModelForCausalLM, AutoTokenizer
import ollama

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TEXT_CHUNK_SIZE = 1000
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

# persona_ids = ["vision_bot", "robotics_bot", "bio_bot", "security_bot", "education_bot", "industry_bot", "theory_bot", "layperson_bot"]
persona_ids = ["layperson_bot"]
device = "mps" if torch.has_mps else "cpu"

def find_relevant_papers(collection, query_text, n_papers=3, cutoff_date=None):
    """
    Find relevant papers using semantic search.
    Args:
        collection: ChromaDB collection
        query_text: Query text for semantic search
        n_papers: Number of papers to return
        cutoff_date: Optional datetime to filter papers published after this date
    """
    query_embedding = embedding_model.encode(query_text).tolist()

    # Build query parameters
    query_params = {
        "query_embeddings": [query_embedding],
        "n_results": n_papers*TEXT_CHUNK_SIZE,
        "include": ["metadatas", "distances"]
    }

    # Add where clause only if we have a cutoff date
    if cutoff_date:
        query_params["where"] = {
            "$and": [
                {"chunk_index": {"$eq": 0}},  # Only look at abstract chunks
                {"published": {"$gt": cutoff_date.isoformat()}}  # Filter by ISO date string
            ]
        }

    # Query the collection
    results = collection.query(**query_params)

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
    """
    Get the most semantically relevant papers published within the last N days.
    
    Args:
        collection: ChromaDB collection containing paper data
        days (int): Number of days to look back for recent papers
        n_papers (int): Number of papers to return
        
    Returns:
        list: List of paper IDs that are both recent and semantically relevant to
             recent advances and breakthroughs in the field
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    query = "novel OR breakthrough OR state-of-the-art OR recent advances"
    return find_relevant_papers(collection, query, n_papers, cutoff_date)

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

    # print(f"Context for {paper_id}: {context}")
    return context, summarizer(
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

def personify_tweet(tweet, persona_id, persona_model):
    bot = BotPersona(persona_id)
    context = bot._get_persona()

    # Tokenize the context and tweet
    prompt = f"Summary: {tweet} Relate tweet to applications of this context: {context}. Don't repeat summary or mention persona, just talk like you're posting a tweet on why you find the paper interesting."
    response = ollama.chat(model=persona_model, messages=[
        {"role": "user", "content": prompt}
    ])

    return response

def evaluate_tweet(context, tweet, eval_model):
    # Tokenize the context and tweet
    prompt = f"""You will be given one summary tweet written for a research paper.

Your task is to rate the tweet on two metrics. Read these instructions carefully and refer back as needed.

Evaluation Criteria:

1. Factual Consistency (1-3): Does the tweet only contain facts supported by the source text?
- 1 (Inconsistent): Major errors or many minor errors
- 2 (Overall consistent): At most one minor error
- 3 (Consistent): All facts supported

2. Engagingness (1-3): Is the tweet interesting to most audiences?
- 1 (Dull): Only interesting to specialists
- 2 (Somewhat interesting): Engages those familiar with the field
- 3 (Interesting): Engages general audiences regardless of expertise

Evaluation Steps:

1. Read the source text and identify its key points.
2. Read the tweet. Check for factual consistency and engagingness.
3. Return two scores as: (Factual Consistency, Engagingness)

Example:

Source Text:
{context}

Summary:
{tweet}

Evaluation Form:
(Factual Consistency, Engagingness):
"""

    response = ollama.chat(model=eval_model, messages=[
        {"role": "user", "content": prompt}
    ])

    return response

def query_and_generate(args):
    summarizer = pipeline("summarization", model=args.model_type)
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "db", args.folder_name)
    client = chromadb.PersistentClient(file_path)
    collection = client.get_collection(name=args.db_name)
    items = collection.get()

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
    
    contexts, tweets = {}, []
    for paper_id in paper_ids:
        # Step 2: Get metadata
        try:
            metadata = get_paper_metadata(collection, paper_id)
        except:
            continue

        # Step 3: Generate summary for this paper
        context, summary = generate_paper_summary(summarizer, collection, paper_id, query)
        if not context or not summary:
            continue
        contexts[paper_id] = context

        # Step 4: Create tweet
        tweets.append((paper_id, generate_tweet(summary, metadata, paper_id)))
    return contexts, tweets

def main():
    parser = argparse.ArgumentParser(description='Generate tweets from paper database')
    parser.add_argument('--persona_model_name', required=False, default="deepseek-r1:1.5b", help='Specify persona model. Default:deepseek-r1:1.5b')
    parser.add_argument('--eval_model_name', required=False, default="deepseek-r1:1.5b", help='Specify eval model. Default:deepseek-r1:1.5b')
    parser.add_argument('--model_type', required=False, default="facebook/bart-large-cnn", help='Specify summarizer model. Default: facebook/bart-large-cnn')
    parser.add_argument('--folder_name', required=True, help='DB folder name (within db)')
    parser.add_argument('--db_name', required=False, default='papers', help='The actual DB name within chroma.sqlite3. Default is "papers"')
    parser.add_argument('--topic', default=None, help='Optional topic to focus on')
    parser.add_argument('--num_papers', type=int, default=3, help='Number of papers to tweet about')
    parser.add_argument('--days', type=int, default=None, help='Focus on papers from last N days')
    args = parser.parse_args()

    contexts, tweets = query_and_generate(args)
    personified_tweets = []

    for persona_id in persona_ids:
        for paper_id, tweet in tweets:
            result = personify_tweet(tweet, persona_id, args.persona_model_name)
            eval = evaluate_tweet(contexts[paper_id], result, args.eval_model_name)
            personified_tweets.append((paper_id, tweet, result, eval))
            
    for paper_id, tweet, result, eval in personified_tweets:
        print(f"Original tweet for {paper_id}: ")
        print(tweet)
        print(f"Generated PERSONIFIED Tweet for {paper_id}:")
        print(result)
        print(f"Evaluation for {paper_id}:")
        print(eval)
        print("\n" + "-"*50 + "\n")

if __name__ == "__main__":
    main()