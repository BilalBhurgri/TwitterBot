# start_here.py 

Generally, you run this like. Make sure to include the extra `/` after any path.

```
python start_here.py --model_name <model_name> --force_reprocess --prefix papers/threads/ --threads
```

The prefix is the path where you download the papers from.
`--force_reprocess` allows you to reprocess papers and ignore the methods that check if a paper was reprocessed. 

## Qwen

Qwen/Qwen3-4B

`python start_here.py --model_name Qwen/Qwen3-4B --force_reprocess --prefix papers/threads/`

`python start_here.py --model_name Qwen/Qwen3-1.7B --force_reprocess --prefix papers/threads/`

## LLAMA 

Prompt not made yet, generate_summary_llama is untested code. generate_tweet_llama isn't written yet.

## OLMO

`python start_here.py --model_name allenai/OLMo-2-0425-1B-Instruct --force_reprocess --prefix papers/threads/ --threads`

`python start_here.py --model_name allenai/OLMo-2-1124-7B-Instruct --force_reprocess --prefix papers/threads/ --threads`

## Mistral 

`python start_here.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --force_reprocess --prefix papers/threads/ --threads`