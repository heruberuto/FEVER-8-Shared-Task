import os
from dataclasses import dataclass
from typing import Dict, List, Union
from classification import ClassificationResult, Classifier
from evidence_generation import EvidenceGenerationResult, EvidenceGenerator
from retrieval import RetrievalResult, Retriever
from averitec import Datapoint
import pickle
import os
from ollama import chat
from ollama import ChatResponse
import json
from tqdm import tqdm
import gc,pickle
from pipeline import Pipeline, MockPipeline, PipelineResult
from evidence_generation import DynamicFewShotBatchedEvidenceGenerator
from classification import NoTiebreakClassifier
gc.collect()
import torch
torch.cuda.empty_cache()

if __name__ == "__main__":
    # Get configuration from environment variables, with fallback defaults
        
    # Paths from env or defaults
    vecstore_path = os.environ.get("VECSTORE_PATH", "data_store/vector_store")
    results_path = os.environ.get("RESULTS_PATH", "data_store/results")
    prompts = os.environ.get("PROMPTS_PATH", "data_store/llm_prompts")
    submission_path = os.environ.get("SUBMISSION_PATH", "data_store/submissions")
    dataset_file = os.environ.get("DATASET_FILE", "data_store/averitec/dev.json")
    train_file = os.environ.get("TRAIN_FILE", "data_store/averitec/train.json")
    PIPELINE_NAME = os.environ.get("SYSTEM_NAME", "aic")
    response_path = os.environ.get("RESPONSE_PATH", "data_store/qwen_responses")

    last_good_response = """```json
{
    \"questions\": [
        {
            \"question\": \"How are you?\",
            \"answer\": \"good\",
            \"source\": \"1\",
            \"answer_type\": \"Extractive\"
        }
    ],
    \"claim_veracity\": {
        \"Supported\": \"1\",
        \"Refuted\": \"5\",
        \"Not Enough Evidence\": \"1\",
        \"Conflicting Evidence/Cherrypicking\": \"1\"
    },
    \"veracity_verdict\": \"Refuted\"
}
```"""
    
    pipeline = MockPipeline(
        dumps=f"{submission_path}/{PIPELINE_NAME}.pkl",
        evidence_generator=DynamicFewShotBatchedEvidenceGenerator(reference_corpus_path=train_file), 
        classifier=NoTiebreakClassifier()
    )
    # load /home/ubuntu/batch_1.jsonl
    with open(prompts+'/batch_1.jsonl', 'r') as f:
        lines = f.readlines()
        data = [json.loads(line) for line in lines]
        
    i=0
    j=len(data)

    if not os.path.exists(response_path):
        os.makedirs(response_path)
    
    new_submission = []   
    for d in tqdm(data):
        print(d["body"]["messages"])
        response: ChatResponse = chat(model='qwen3-custom', messages=d["body"]["messages"])
        with open(f'{response_path}/{i}.txt', 'w') as f:
            f.write(response['message']['content'])
            
        current_response = response['message']['content']
        
        try:
            new_result = pipeline.evidence_generator.update_pipeline_result(pipeline.dumps[len(new_submission)], current_response, pipeline.classifier)
            new_submission.append(new_result.to_submission())
            last_good_response = current_response
        except:
            new_result = pipeline.evidence_generator.update_pipeline_result(pipeline.dumps[len(new_submission)], last_good_response, pipeline.classifier)
            new_submission.append(new_result.to_submission())
            
        if True:
            with open(f"{submission_path}/{PIPELINE_NAME}.json", "w") as f:
                json.dump(new_submission, f, indent=4) 
                
    with open(f"{submission_path}/{PIPELINE_NAME}.json", "w") as f:
        json.dump(new_submission, f, indent=4) 