import tqdm
import argparse
import torch
import transformers
import json
from vllm import LLM, SamplingParams
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional

LABEL = [
    "Supported",
    "Refuted",
    "Not Enough Evidence",
    "Conflicting Evidence/Cherrypicking",
]

def format_time(seconds: float) -> str:
    """Format time duration nicely."""
    return str(timedelta(seconds=round(seconds)))

def get_label_from_output(output: str) -> Optional[str]:
    """Extract label from model output."""
    if "Not Enough Evidence" in output:
        return "Not Enough Evidence"
    elif any(x in output for x in ["Conflicting Evidence/Cherrypicking", "Cherrypicking", "Conflicting Evidence"]):
        return "Conflicting Evidence/Cherrypicking"
    elif any(x in output for x in ["Supported", "supported"]):
        return "Supported"
    elif any(x in output for x in ["Refuted", "refuted"]):
        return "Refuted"
    return None

def prepare_prompts(examples: List[Dict], tokenizer, model_name: str) -> List[str]:
    """Prepare prompts for batch processing."""
    base_prompt = "Your task is to predict the verdict of a claim based on the provided question-answer pair evidence. Choose from the labels: 'Supported', 'Refuted', 'Not Enough Evidence', 'Conflicting Evidence/Cherrypicking'. Disregard irrelevant question-answer pairs when assessing the claim. Justify your decision step by step using the provided evidence and select the appropriate label."
    
    prepared_inputs = []
    for example in examples:
        example["input_str"] = base_prompt + "\n\nClaim: " + example["claim"] + "\n\n" + "\n\n".join(
            [f"Q{i+1}: {qa['question']}\nA{i+1}: {qa['answer']}" for i, qa in enumerate(example["evidence"])]
        )

        if "OLMo" in model_name:
            input_ids = [example["input_str"]]
        else:
            messages = [{"role": "user", "content": example["input_str"]}]
            input_ids = tokenizer.apply_chat_template(messages, tokenize=False)
        
        prepared_inputs.append(input_ids)
    
    return prepared_inputs

def main(args):
    script_start = time.time()
    start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Script started at: {start_time}")

    # Load data
    data_load_start = time.time()
    try:
        with open(args.target_data) as f:
            examples = json.load(f)
    except:
        examples = []
        with open(args.target_data) as f:
            for line in f:
                examples.append(json.loads(line))
    print(f"Data loading took: {format_time(time.time() - data_load_start)}")
    print(f"Total examples to process: {len(examples)}")

    # Initialize model and tokenizer
    model_start = time.time()
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model)
    
    gpu_counts = torch.cuda.device_count()
    print(f"Using {gpu_counts} GPU{'s' if gpu_counts > 1 else ''}")

    llm = LLM(
        model=args.model,
        tensor_parallel_size=gpu_counts,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=True,
        trust_remote_code=True,
        # dtype="bfloat16"
    )
    print(f"Model initialization took: {format_time(time.time() - model_start)}")

    sampling_params = SamplingParams(
        temperature=0.9,
        top_p=0.7,
        top_k=1,
        skip_special_tokens=False,
        max_tokens=512,
        stop=['<|endoftext|>', '</s>', '<|im_end|>', '[INST]', '[/INST]','<|eot_id|>','<|end|>']
    )

    # Process in batches
    batch_size = args.batch_size
    predictions = []
    processing_start = time.time()
    
    with torch.no_grad():
        for batch_idx in range(0, len(examples), batch_size):
            batch_start = time.time()
            batch_end = min(batch_idx + batch_size, len(examples))
            current_batch = examples[batch_idx:batch_end]
            
            # Prepare batch prompts
            batch_inputs = prepare_prompts(current_batch, tokenizer, args.model)
            
            # Process batch
            outputs = llm.generate(batch_inputs, sampling_params)
            
            # Process outputs
            for example, output in zip(current_batch, outputs):
                output_text = output.outputs[0].text.strip()
                label = get_label_from_output(output_text)
                
                # Retry with different sampling if no label found
                retry_count = 0
                while label is None and retry_count < 3:
                    retry_count += 1
                    retry_params = SamplingParams(
                        temperature=0.9,
                        top_p=0.7,
                        top_k=2,
                        skip_special_tokens=False,
                        max_tokens=512,
                        stop=['<|endoftext|>', '</s>', '<|im_end|>', '[INST]', '[/INST]','<|eot_id|>','<|end|>']
                    )
                    retry_output = llm.generate(example["input_str"], retry_params)
                    output_text = retry_output[0].outputs[0].text.strip()
                    label = get_label_from_output(output_text)
                    if label is None:
                        print(f"Retry {retry_count}: Could not find label in output.")
                        print(output_text)

                json_data = {
                    "claim_id": example["claim_id"],
                    "claim": example["claim"],
                    "evidence": example["evidence"],
                    "pred_label": label or "Not Enough Evidence",  # fallback if still no label
                    "llm_output": output_text,
                }
                predictions.append(json_data)
            
            batch_time = time.time() - batch_start
            print(f"Processed batch {batch_idx}-{batch_end} in {batch_time:.2f}s")

    # Save predictions
    with open(args.output_file, "w", encoding="utf-8") as output_file:
        json.dump(predictions, output_file, ensure_ascii=False, indent=4)

    # Calculate and display timing information
    total_time = time.time() - script_start
    processing_time = time.time() - processing_start
    end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print("\nTiming Summary:")
    print(f"Start time: {start_time}")
    print(f"End time: {end_time}")
    print(f"Total runtime: {format_time(total_time)}")
    print(f"Data loading time: {format_time(data_load_start - script_start)}")
    print(f"Model initialization time: {format_time(processing_start - model_start)}")
    print(f"Processing time: {format_time(processing_time)}")
    print(f"Average time per example: {processing_time / len(examples):.2f}s")
    print(f"Processing speed: {len(examples) / processing_time:.2f} examples per second")
    print(f"\nResults written to: {args.output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default="humane-lab/Meta-Llama-3.1-70B-HerO")
    parser.add_argument("-i", "--target_data", default="data_store/averitec/dev.json")
    parser.add_argument("-o", "--output_file", default="data_store/dev_veracity_prediction.json")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for processing")
    args = parser.parse_args()    
    
    main(args)