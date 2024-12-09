import argparse
import json
import os
import time
import numpy as np
import pandas as pd
import nltk
from rank_bm25 import BM25Okapi
from multiprocessing import Pool, cpu_count, Manager, Lock
from functools import partial
import heapq
from threading import Thread, Event
import queue

def combine_all_sentences(knowledge_file):
    sentences, urls = [], []
    
    with open(knowledge_file, "r", encoding="utf-8") as json_file:
        for i, line in enumerate(json_file):
            data = json.loads(line)
            sentences.extend(data["url2text"])
            urls.extend([data["url"] for _ in range(len(data["url2text"]))])
    return sentences, urls, i + 1

def remove_duplicates(sentences, urls):
    df = pd.DataFrame({"document_in_sentences":sentences, "sentence_urls":urls})
    df['sentences'] = df['document_in_sentences'].str.strip().str.lower()
    df = df.drop_duplicates(subset="sentences").reset_index()
    return df['document_in_sentences'].tolist(), df['sentence_urls'].tolist()
                
def retrieve_top_k_sentences(query, document, urls, top_k):
    tokenized_docs = [nltk.word_tokenize(doc) for doc in document]
    bm25 = BM25Okapi(tokenized_docs)
    
    scores = bm25.get_scores(nltk.word_tokenize(query))
    top_k_idx = np.argsort(scores)[::-1][:top_k]

    return [document[i] for i in top_k_idx], [urls[i] for i in top_k_idx]

def process_single_example(idx, example, args, result_queue, counter, lock):
    try:
        with lock:
            current_count = counter.value + 1
            counter.value = current_count
            print(f"\nProcessing claim {idx}... Progress: {current_count} / {args.total_examples}")
        
        start_time = time.time()
        
        # Load the knowledge store for this example
        document_in_sentences, sentence_urls, num_urls_this_claim = combine_all_sentences(
            os.path.join(args.knowledge_store_dir, f"{idx}.json")
        )
        
        print(f"Obtained {len(document_in_sentences)} sentences from {num_urls_this_claim} urls.")
        
        # Remove duplicate sentences in knowledge store
        document_in_sentences, sentence_urls = remove_duplicates(document_in_sentences, sentence_urls)
        
        # Retrieve top_k sentences with bm25
        query = example["claim"] + " " + " ".join(example['hypo_fc_docs'])
        top_k_sentences, top_k_urls = retrieve_top_k_sentences(
            query, document_in_sentences, sentence_urls, args.top_k
        )
        
        processing_time = time.time() - start_time
        print(f"Top {args.top_k} retrieved. Time elapsed: {processing_time:.2f}s")

        result = {
            "claim_id": idx,
            "claim": example["claim"],
            f"top_{args.top_k}": [
                {"sentence": sent, "url": url}
                for sent, url in zip(top_k_sentences, top_k_urls)
            ],
            "hypo_fc_docs": example['hypo_fc_docs']
        }
        
        # Put the result in the queue with its index for ordering
        result_queue.put((idx, result))
        
        return True
    except Exception as e:
        print(f"Error processing example {idx}: {str(e)}")
        result_queue.put((idx, None))
        return False

def writer_thread(output_file, result_queue, total_examples, stop_event):
    """Thread that writes results to file in order."""
    next_index = 0
    pending_results = []
    
    with open(output_file, "w", encoding="utf-8") as f:
        while not (stop_event.is_set() and result_queue.empty()):
            try:
                # Get result with timeout to check stop_event periodically
                idx, result = result_queue.get(timeout=1)
                
                # Add to pending results
                if result is not None:  # Skip failed results
                    heapq.heappush(pending_results, (idx, result))
                
                # Write any results that are next in sequence
                while pending_results and pending_results[0][0] == next_index:
                    _, result_to_write = heapq.heappop(pending_results)
                    f.write(json.dumps(result_to_write, ensure_ascii=False) + "\n")
                    f.flush()
                    next_index += 1
                    
            except queue.Empty:
                continue

def main(args):
    # Load target examples
    with open(args.target_data, "r", encoding="utf-8") as json_file:
        target_examples = json.load(json_file)

    if args.end == -1:
        args.end = len(target_examples)
    
    print(f"Total examples to process: {args.end - args.start}")

    # Prepare the list of examples to process
    files_to_process = list(range(args.start, args.end))
    examples_to_process = [(idx, target_examples[idx]) for idx in files_to_process]
    
    # Determine number of workers
    num_workers = min(args.workers if args.workers > 0 else cpu_count(), len(files_to_process))
    print(f"Using {num_workers} workers to process {len(files_to_process)} examples")

    # Create a manager for shared objects
    with Manager() as manager:
        counter = manager.Value('i', 0)
        lock = manager.Lock()  # Create a lock for synchronization
        args.total_examples = len(files_to_process)
        
        # Create a queue for results
        result_queue = manager.Queue()
        
        # Create and start the writer thread
        stop_event = Event()
        writer = Thread(
            target=writer_thread,
            args=(args.json_output, result_queue, len(files_to_process), stop_event)
        )
        writer.start()

        # Create partial function with fixed arguments
        process_func = partial(
            process_single_example,
            args=args,
            result_queue=result_queue,
            counter=counter,
            lock=lock  # Pass the lock to the processing function
        )
        
        # Process examples in parallel
        with Pool(num_workers) as pool:
            results = pool.starmap(process_func, examples_to_process)
        
        # Signal writer thread to stop and wait for it to finish
        stop_event.set()
        writer.join()
        
        # Calculate success rate
        successful = sum(1 for r in results if r)
        print(f"\nSuccessfully processed {successful} out of {len(files_to_process)} examples")
        print(f"Results written to {args.json_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get top 10000 sentences with BM25 in the knowledge store using parallel processing."
    )
    parser.add_argument(
        "-k",
        "--knowledge_store_dir",
        type=str,
        default="data_store/knowledge_store",
        help="The path of the knowledge_store_dir containing json files with all the retrieved sentences.",
    )
    parser.add_argument(
        "--target_data",
        type=str,
        default="data_store/hyde_fc.json",
        help="The path of the file that stores the claim.",
    )
    parser.add_argument(
        "-o",
        "--json_output",
        type=str,
        default="data_store/dev_retrieval_top_k.json",
        help="The output dir for JSON files to save the top 100 sentences for each claim.",
    )
    parser.add_argument(
        "--top_k",
        default=10000,
        type=int,
        help="How many documents should we pick out with BM25.",
    )
    parser.add_argument(
        "-s",
        "--start",
        type=int,
        default=0,
        help="Starting index of the files to process.",
    )
    parser.add_argument(
        "-e",
        "--end",
        type=int,
        default=-1,
        help="End index of the files to process.",
    )
    parser.add_argument(
        "-w",
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes (default: number of CPU cores)",
    )

    args = parser.parse_args()
    main(args)