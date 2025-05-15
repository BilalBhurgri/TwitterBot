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

## GCP VM

In theory, one user installing necessary packages and cuda toolkit should make them available to all users of the vm. 
If you try `nvidia-smi` and it isn't recognized, then you might want to try these installation steps: 

1. https://cloud.google.com/compute/docs/gpus/install-drivers-gpu
