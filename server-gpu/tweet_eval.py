from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import bert_score
import json
import boto3
from dotenv import load_dotenv
import os
from botocore.exceptions import ClientError
import evaluate
from statistics import mean

load_dotenv()
s3 = boto3.client('s3', region_name='us-west-1')

rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def eval_human(label):
    with open("data/all_threads.json", "r") as f:
        data = json.load(f)
    if label == "full":
        threads = [(item["paper"].split('/')[-1], ''.join(item["tweets"])) for item in data["threads"]]
    else:
        threads = [(item["paper"].split('/')[-1], item["tweets"][0]) for item in data["threads"]]
    references = []
    predictions = []
    rouge_results = []

    for paper_id, tweet in threads:
        print(f"Curr paper_id = {paper_id}")
        paper_text = ''
        try:
            obj = s3.get_object(Bucket=os.environ.get('BUCKET_NAME'), Key=f'papers/threads/{paper_id}.txt')
            paper_text = obj['Body'].read().decode('utf-8')
            rouge_result = rouge.compute(predictions=[tweet], references=[paper_text])
            rouge_results.append({"paper_id": paper_id, **rouge_result})
            references.append(paper_text)
            predictions.append(tweet)
        except ClientError as e:
            print(f"Error getting object papers/{paper_id}.txt: {e}")
            continue
    
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")
    combined_results = []

    for i in range(len(rouge_results)):
        combined = {
            "paper_id": rouge_results[i]["paper_id"],
            "rouge1": rouge_results[i]["rouge1"],
            "rouge2": rouge_results[i]["rouge2"],
            "rougeL": rouge_results[i]["rougeL"],
            "rougeLsum": rouge_results[i]["rougeLsum"],
            "bertscore_precision": bert_results["precision"][i],
            "bertscore_recall": bert_results["recall"][i],
            "bertscore_f1": bert_results["f1"][i],
            "len": len(predictions[i])
        }
        combined_results.append(combined)
    
    # Get all metric names except 'index'
    metric_keys = [k for k in combined_results[0] if k != "paper_id"]

    # Compute mean for each metric
    metric_means = {
        metric: mean(result[metric] for result in combined_results)
        for metric in metric_keys
    }

    results = {"method": f"human_{label}", "scores": combined_results, "means": metric_means}
    with open(f"eval_results/human_{label}.json", "w") as f:
        json.dump(results, f, indent=2)

def eval_model(num_bot, model, label):
    with open("data/all_threads.json", "r") as f:
        data = json.load(f)
    threads = [item["paper"].split('/')[-1] for item in data["threads"]]
    references = []
    predictions = []
    rouge_results = []

    for paper_id in threads:
        print(f"Curr paper_id = {paper_id}")
        paper_text = ''
        try:
            obj = s3.get_object(Bucket=os.environ.get('BUCKET_NAME'), Key=f'papers/threads/{paper_id}.txt')
            paper_text = obj['Body'].read().decode('utf-8')
            tweet = ''
            if label == "thread":
                path = f"threads/{model}/bot{num_bot}/{paper_id}.json"
            else:
                path = f"downloads/thread_downloads/{model}/bot{num_bot}/{paper_id}.json"

            if os.path.exists(path):
                with open(path, "r") as f:
                    data = json.load(f)
                    if (label == "bad_summary"):
                        if data["best_summary_idx"] < 0:
                            continue
                        tweet = data["all_summaries"][1 - data["best_summary_idx"]]
                    elif label == "thread":
                        tweet = ''.join(data["thread"])
                    else:
                        tweet = data[label]
                    rouge_result = rouge.compute(predictions=[tweet], references=[paper_text])
                    rouge_results.append({"paper_id": paper_id, **rouge_result})
                    predictions.append(tweet)
                references.append(paper_text)

        except ClientError as e:
            print(f"Error getting object papers/{paper_id}.txt: {e}")
            continue
    
    bert_results = bertscore.compute(predictions=predictions, references=references, lang="en")
    combined_results = []

    for i in range(len(rouge_results)):
        combined = {
            "paper_id": rouge_results[i]["paper_id"],
            "rouge1": rouge_results[i]["rouge1"],
            "rouge2": rouge_results[i]["rouge2"],
            "rougeL": rouge_results[i]["rougeL"],
            "rougeLsum": rouge_results[i]["rougeLsum"],
            "bertscore_precision": bert_results["precision"][i],
            "bertscore_recall": bert_results["recall"][i],
            "bertscore_f1": bert_results["f1"][i],
            "len": len(predictions[i])
        }
        combined_results.append(combined)
    
    # Get all metric names except 'index'
    metric_keys = [k for k in combined_results[0] if k != "paper_id"]
    # Compute mean for each metric
    metric_means = {
        metric: mean(result[metric] for result in combined_results)
        for metric in metric_keys
    }
    results = {"method": f"{model}_bot{num_bot}_{label}", "scores": combined_results, "means": metric_means}

    with open(f"eval_results/{model}/bot{num_bot}/{label}.json", "w") as f:
        json.dump(results, f, indent=2)

def main():
    eval_model(0, "Llama-3.1-8B-Instruct", "thread")

if __name__ == "__main__":
    main()