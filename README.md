# TwitterBot


## Setup

In the project root folder:

1.  Make .env file with api keys
2. `python -m venv .venv` for first time setup.
3. `source .venv/bin/activate` 
4. `pip install -r requirements.txt` (might take a minute)

## Testing

In the project root folder:

Tweet generation: `python -m try_models.query --name NAME [--topic TOPIC] [--num_papers NUM_PAPERS] [--days DAYS]`
- use `papers2` for `--name`
- to retrieve papers published recently, use `--days`
- to screen abstracts based on a topic, use `--topic`
- to do full text reviews, use neither `--days` nor `--topic`
- default value of `--num_papers`: 3

Posting: `python -m bot.bot --name NAME [--topic TOPIC] [--num_papers NUM_PAPERS] [--days DAYS] [--post]`
- same arguments for tweet generation
- this makes a mock post on default, use `--post` to actually post on Twitter


# Overview of scripts

## simple-update-db.py 

As of May 19:
- Embeddings are made out of text chunks of size 1000. This was chosen because a chunk size between 1000-2000 is good for summarization, while smaller sizes such as 250-500 are better for extracting specific details. 
- Embedding model is "all-MiniLM-L6-v2". This is kind of basic. Here's an embedding leaderboard: https://huggingface.co/spaces/mteb/leaderboard. You can experiment with this by creating chromadbs with different embedding models (put the model in the name). Embedding models do have a significant impact on RAG's retrieval quality, and higher quality embeddings preserve the meaning of the text better. Note that embedding models may need different text chunk sizes!

Arguments:
```python
parser = argparse.ArgumentParser(description='Create/update Chroma DB')
parser.add_argument('--name', required=True, help='DB name')
parser.add_argument('--input', required=True, help='File with arXiv URLs')
parser.add_argument('--embedding_model', required=True, help='Embedding model')
parser.add_argument('--text_chunk_size', default=1024, help='Text chunk size')
```

Command:
```
python simple-update-db.py --name namehere --input papers.txt --embedding_model all-MiniLM-L6-v2 --text_chunk_size 1000
```
`papers.txt` is the text file where all the paper arxiv links are stored.

You know this has worked when you don't see "Failed to process" errors, and check the txts inside the pdfs folder.

### Requirement for this to work: GROBID 

1. See this, install CRF-only image. https://grobid.readthedocs.io/en/latest/Grobid-docker/ 
2. `sudo docker run --rm --init --ulimit core=0 -p 8070:8070 lfoppiano/grobid:0.8.2`


## query_full_paper_verbose.py

May 19: only works with Qwen3-1.7B, maybe also Qwen3-4B. mistralai/Mistral-7B-Instruct-v0.1 isn't outputting legible results currently. 
To add more models, add more functions for each one. Get familiar with the Huggingface generator API, and read the model pages on Huggingface.

May 20: added evaluation logic. NOTE: input should be txt, output paths should be specific file names

Arguments:
```python
parser = argparse.ArgumentParser(description='Generate paper summary using the specified model')
parser.add_argument('--paper_path', required=True, help='Path to the paper PDF file')
parser.add_argument('--output_path', default=None, help='Path to save the summary')
parser.add_argument('--eval_path', default=None, help='Path to save the evaluation')
parser.add_argument('--model_name', default="Qwen/Qwen3-1.7B", help='Model to use (default: Qwen/Qwen3-1.7B)')
```

Command: 
```python
python query_full_paper_verbose.py --model_name Qwen/Qwen3-4B, Qwen/Qwen3-1.7B. 
```

## Other files

get_paper_xml.py, parse_paper.py (doesn't remove math that are outside of formulas), parse_paper_remove_math.py (does remove math). 

## Connecting to GCP VM

In theory, one user installing necessary packages and cuda toolkit should make them available to all users of the vm. 
If you try `nvidia-smi` and it isn't recognized, then you might want to try these installation steps: 

1. https://cloud.google.com/compute/docs/gpus/install-drivers-gpu

Use internal IP addresses and see if you can ping and run scripts on the vm-with-gpu-2. 