import os
import argparse
import time
import json
import nltk
from rank_bm25 import BM25Okapi
import numpy as np
import torch
from vllm import LLM, SamplingParams
from datetime import datetime, timedelta
from itertools import islice


def download_nltk_data(package_name, download_dir='nltk_data'):
    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)
    
    # Set NLTK data path
    nltk.data.path.append(download_dir)
    
    try:
        # Try to find the resource
        nltk.data.find(f'tokenizers/{package_name}')
        print(f"Package '{package_name}' is already downloaded")
    except LookupError:
        # If resource isn't found, download it
        print(f"Downloading {package_name}...")
        nltk.download(package_name, download_dir=download_dir)
        print(f"Successfully downloaded {package_name}")

def format_time(seconds):
    """Format time duration nicely."""
    return str(timedelta(seconds=round(seconds)))

def claim2prompts(example): 
    claim = example["claim"]
    claim_str = "Example [NUMBER]:||Claim: " + claim + "||Evidence: "

    for question in example["questions"]:
        q_text = question["question"].strip()
        if len(q_text) == 0:
            continue

        if not q_text[-1] == "?":
            q_text += "?"

        answer_strings = []

        for a in question["answers"]:
            if a["answer_type"] in ["Extractive", "Abstractive"]:
                answer_strings.append(a["answer"])
            if a["answer_type"] == "Boolean":
                answer_strings.append(a["answer"]  + ", because " + a["boolean_explanation"].lower().strip())

        for a_text in answer_strings:
            if not a_text[-1] in [".", "!", ":", "?"]:
                a_text += "."

            prompt_lookup_str = a_text
            this_q_claim_str = claim_str + a_text.strip() + "||Question: " + q_text 
            yield (prompt_lookup_str, this_q_claim_str.replace("\n", " ").replace("||", "\n")[:1500])

def main(args):
    script_start = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Script started at: {start_time}")
    print(f"Loading model: {args.model}")


    download_nltk_data('punkt')
    download_nltk_data('punkt_tab')

    # Load and prepare reference corpus
    corpus_start = time.time()
    with open(args.reference_corpus, "r", encoding="utf-8") as json_file:
        train_examples = json.load(json_file)

    prompt_corpus, tokenized_corpus = [], []
    for example in train_examples:
        for lookup_str, prompt in claim2prompts(example):
            entry = nltk.word_tokenize(lookup_str)
            tokenized_corpus.append(entry)
            prompt_corpus.append(prompt)

    prompt_bm25 = BM25Okapi(tokenized_corpus)
    print(f"Reference corpus processed in: {format_time(time.time() - corpus_start)}")
    
    # Initialize vLLM with optimized settings
    gpu_count = torch.cuda.device_count()
    print(f"Using {gpu_count} GPU{'s' if gpu_count > 1 else ''}")
    
    model_start = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=gpu_count,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        trust_remote_code=True,
        # dtype="half",
    )
    llm.get_tokenizer().pad_token = "<|end_of_text|>"
    print(f"Model loaded in: {format_time(time.time() - model_start)}")
    
    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.9,
        top_k=1,
        skip_special_tokens=False,
        max_tokens=512,
        stop=['<|end_of_text|>', '</s>', '<|im_end|>', '[INST]', '[/INST]','<|eot_id|>','<|end|>','<|endoftext|>']
    )

    processing_start = time.time()

    # Load target data
    target_examples = []
    with open(args.top_k_target_knowledge, "r", encoding="utf-8") as json_file:
        for line in json_file:
            target_examples.append(json.loads(line))

    if args.end == -1:
        args.end = len(target_examples)
    print(f"Processing {args.end} examples")

    # Process in batches
    with torch.no_grad():
        with open(args.output_questions, "w", encoding="utf-8") as output_file:
            for idx in range(0, args.end, args.batch_size):
                batch_end = min(idx + args.batch_size, args.end)
                current_batch = target_examples[idx:batch_end]
                print(f"\nProcessing batch {idx}-{batch_end}...")
                
                for example in current_batch:
                    batch_start = time.time()
                    claim = example["claim"]
                    claim_id = example["claim_id"]
                    top_k_sentences_urls = example[f"top_{args.top_k}"]
                    
                    batch_prompts = []
                    batch_metadata = []
                    
                    # Prepare all prompts for current example
                    for sentences_urls in top_k_sentences_urls:
                        prompt_lookup_str = sentences_urls["sentence"]
                        url = sentences_urls["url"]
                        
                        prompt_s = prompt_bm25.get_scores(nltk.word_tokenize(prompt_lookup_str))
                        prompt_n = 10
                        prompt_top_n = np.argsort(prompt_s)[::-1][:prompt_n]
                        prompt_docs = [prompt_corpus[i] for i in prompt_top_n]
                        
                        temp_prompt = "\n\n".join(prompt_docs)
                        for k in range(1, temp_prompt.count("[NUMBER]")+1):
                            temp_prompt = temp_prompt.replace("[NUMBER]", f"{k}", 1)
                            
                        claim_prompt = "Your task is to generate a question based on the given claim and evidence. The question should clarify the relationship between the evidence and the claim\n\n"
                        evidence = prompt_lookup_str.replace("\n", " ")
                        full_prompt = claim_prompt + temp_prompt + "\n\nNow, generate a question that links the following claim and evidence:" + f"\n\nClaim: {claim}" + f"\nEvidence: {evidence}"
                        
                        if "OLMo" in args.model:
                            inputs = [full_prompt]
                        else:
                            messages = [{"role":"user", "content":full_prompt}]
                            inputs = llm.get_tokenizer().apply_chat_template(messages, tokenize=False)
                            inputs += "<|start_header_id|>assistant<|end_header_id|>\n\nQuestion: "
                            
                        batch_prompts.append(inputs)
                        batch_metadata.append((url, prompt_lookup_str))
                    
                    # Process batch
                    outputs = llm.generate(batch_prompts, sampling_params)
                    
                    # Process outputs
                    evidence = []
                    for output, (url, sent) in zip(outputs, batch_metadata):
                        question = output.outputs[0].text.strip().split("?")[0].replace("\n", " ") + "?"
                        evidence.append({
                            "question": question,
                            "answer": sent,
                            "url": url
                        })
                    
                    # Write results
                    json_data = {
                        "claim_id": claim_id,
                        "claim": claim,
                        "evidence": evidence
                    }
                    output_file.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                    output_file.flush()
                    
                    batch_time = time.time() - batch_start
                    print(f"Processed example {claim_id}. Time elapsed: {batch_time:.2f}s")

    # Calculate and display timing information
    total_time = time.time() - script_start
    processing_time = time.time() - processing_start
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\nTiming Summary:")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total runtime: {format_time(total_time)}")
    print(f"Setup time: {format_time(processing_start - script_start)}")
    print(f"Processing time: {format_time(processing_time)}")
    print(f"Results written to: {args.output_questions}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Use a prompt to generate questions that could be answered by top-k retrieved evidence. Output generated questions.")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--reference_corpus", default="data_store/averitec/train.json")
    parser.add_argument(
        "-i",
        "--top_k_target_knowledge",
        default="data_store/dev_reranking_top_k.json",
        help="Directory where the sentences for the scraped data is saved.",
    )
    parser.add_argument(
        "-o",
        "--output_questions",
        default="data_store/dev_top_k_qa.json",
        help="Directory where the sentences for the scraped data is saved.",
    )
    parser.add_argument(
        "--top_k",
        default=10,
        type=int
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of examples to process in each batch"
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=-1
    )
    
    args = parser.parse_args()
    main(args)