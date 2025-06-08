#!/bin/bash

python start_here.py --model_name mistralai/Mistral-7B-Instruct-v0.3 --force_reprocess --prefix papers/threads/ --upload_prefix results-eval-threads
python start_here.py --model_name meta-llama/Llama-3.1-8B-Instruct --force_reprocess --prefix papers/threads/ --upload_prefix results-eval-threads