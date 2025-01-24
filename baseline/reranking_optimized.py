import os
import torch
import gc
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import argparse
import time
from datetime import datetime, timedelta
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def encode_text(model, tokenizer, texts, batch_size=8, max_length=512):
    """Encode texts to embeddings using AutoModel"""
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        
        # Tokenize
        encoded_input = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        ).to(model.device)
        
        # Compute token embeddings
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                model_output = model(**encoded_input)
                # Use mean pooling
                attention_mask = encoded_input['attention_mask']
                token_embeddings = model_output[0]  # First element contains token embeddings
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                all_embeddings.append(embeddings.cpu().numpy())
        
        # Clear some memory
        if i % (batch_size * 4) == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return np.vstack(all_embeddings)

def compute_similarity(emb1, emb2):
    """Compute cosine similarity between embeddings"""
    return np.dot(emb1, emb2.T) / (
        np.linalg.norm(emb1, axis=1).reshape(-1, 1) * 
        np.linalg.norm(emb2, axis=1).reshape(1, -1)
    )

def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery: {query}'

def preprocess_sentences(sentence1, sentence2):
    vectorizer = TfidfVectorizer().fit_transform([sentence1, sentence2])
    vectors = vectorizer.toarray()
    
    cosine_sim = cosine_similarity(vectors)
    similarity_score = cosine_sim[0][1]
    return similarity_score

def remove_trailing_special_chars(text):
    return re.sub(r'[\W_]+$', '', text)

def remove_special_chars_except_spaces(text):
    return re.sub(r'[^\w\s]+', '', text)

def select_top_k(claim, results, top_k):
    '''
    remove sentence of similarity claim
    '''
    dup_check = set()
    top_k_sentences_urls = []
    
    i = 0
    print(results)
    claim = remove_special_chars_except_spaces(claim).lower()
    while len(top_k_sentences_urls) < top_k and i < len(results):
        print(i)
        sentence = remove_special_chars_except_spaces(results[i]['sentence']).lower()
        
        if sentence not in dup_check:
            if preprocess_sentences(claim, sentence) > 0.97:
                dup_check.add(sentence)
                continue
            
            if claim in sentence:
                if len(claim) / len(sentence) > 0.92:
                    dup_check.add(sentence)
                    continue 
            
            top_k_sentences_urls.append({
                'sentence': results[i]['sentence'],
                'url': results[i]['url']}
            )
        i += 1
        
    return top_k_sentences_urls

def format_time(seconds):
    """Format time duration nicely."""
    return str(timedelta(seconds=round(seconds)))


def compute_embeddings_batched(model, texts, batch_size=8):
    """Compute embeddings in smaller batches to manage memory"""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):  # Use bfloat16
            emb = model.encode(batch, batch_size=len(batch), show_progress_bar=False)
            all_embeddings.append(emb)
        
        # Clear some memory
        if i % (batch_size * 4) == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    return np.vstack(all_embeddings)

def main(args):
    script_start = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Script started at: {start_time}")
    
    device = "cuda" if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model = AutoModel.from_pretrained(
        "Salesforce/SFR-Embedding-2_R",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-2_R")
    
    # Load target examples
    target_examples = []
    with open(args.target_data, "r", encoding="utf-8") as json_file:
        for i, line in enumerate(json_file):
            try:
                example = json.loads(r"{}".format(line))
                target_examples.append(example)
            except:
                print(f"CURRENT LINE broken {i}")
    
    if args.end == -1:
        args.end = len(target_examples)
    
    files_to_process = list(range(args.start, args.end))
    total = len(files_to_process)
    
    task = 'Given a web search query, retrieve relevant passages that answer the query'
    
    with open(args.json_output, "w", encoding="utf-8") as output_json:
        done = 0
        for idx, example in enumerate(target_examples):
            if idx in files_to_process:
                print(f"Processing claim {example['claim_id']}... Progress: {done + 1} / {total}")
                
                claim = example['claim']
                query = [get_detailed_instruct(task, claim)] + [
                    get_detailed_instruct(task, le) 
                    for le in example['hypo_fc_docs'] 
                    if len(le.strip()) > 0
                ]
                query_length = len(query)
                sentences = [sent['sentence'] for sent in example[f'top_{5000}']][:args.retrieved_top_k]
                
                st = time.time()
                try:
                    # Process query embeddings
                    query_embeddings = encode_text(model, tokenizer, query, batch_size=4)
                    avg_emb_q = np.mean(query_embeddings, axis=0)
                    hyde_vector = avg_emb_q.reshape((1, -1))
                    
                    # Process sentence embeddings in smaller chunks
                    sentence_embeddings = encode_text(
                        model,
                        tokenizer,
                        sentences, 
                        batch_size=args.batch_size
                    )
                    
                    # Compute similarities in chunks to save memory
                    chunk_size = 1000
                    all_scores = []
                    for i in range(0, len(sentence_embeddings), chunk_size):
                        chunk = sentence_embeddings[i:i + chunk_size]
                        chunk_scores = compute_similarity(hyde_vector, chunk)[0]
                        all_scores.extend(chunk_scores)
                    
                    scores = np.array(all_scores)
                    top_k_idx = np.argsort(scores)[::-1]
                    results = [example['top_5000'][i] for i in top_k_idx]
                    top_k_sentences_urls = select_top_k(claim, results, args.top_k)
                    
                    print(f"Top {args.top_k} retrieved. Time elapsed: {time.time() - st:.2f}s")
                    
                    json_data = {
                        "claim_id": example['claim_id'],
                        "claim": claim,
                        f"top_{args.top_k}": top_k_sentences_urls
                    }
                    output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                    output_json.flush()
                    
                except RuntimeError as e:
                    print(f"Error processing claim {example['claim_id']}: {e}")
                    continue
                
                done += 1
    
    # Calculate and display timing information
    total_time = time.time() - script_start
    avg_time = total_time / total
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\nTiming Summary:")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total runtime: {format_time(total_time)} (HH:MM:SS)")
    print(f"Average time per example: {avg_time:.2f} seconds")
    print(f"Processing speed: {total / total_time:.2f} examples per second")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_data", default="data_store/dev_retrieval_top_k.json")
    parser.add_argument("--retrieved_top_k", type=int, default=5000)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("-o", "--json_output", type=str, default="data_store/dev_reranking_top_k.json")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("-s", "--start", type=int, default=0)
    parser.add_argument("-e", "--end", type=int, default=-1)
    args = parser.parse_args()
    
    main(args)