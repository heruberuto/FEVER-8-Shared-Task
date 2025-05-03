import os
import json
import pickle
from tqdm import tqdm
from retrieval import MmrFaissRetriever
from evidence_generation import DynamicFewShotBatchedEvidenceGenerator
from classification import DefaultClassifier, NoTiebreakClassifier
from pipeline import Pipeline, MockPipeline
from averitec import Datapoint

# Paths from env or defaults
vecstore_path = os.environ.get("VECSTORE_PATH", "data_store/vector_store")
results_path = os.environ.get("RESULTS_PATH", "data_store/results")
prompts = os.environ.get("PROMPTS_PATH", "data_store/llm_prompts")
submission_path = os.environ.get("SUBMISSION_PATH", "data_store/submissions")
dataset_file = os.environ.get("DATASET_FILE", "data_store/averitec/test_2025.json")
train_file = os.environ.get("TRAIN_FILE", "data_store/averitec/train.json")
PIPELINE_NAME = os.environ.get("SYSTEM_NAME", "mmr+qwen3-dfewshot-atype")
response_path = os.environ.get("RESPONSE_PATH", "data_store/qwen_responses")

if __name__ == "__main__":
    os.makedirs(submission_path, exist_ok=True)

    datapoints = []
    with open(dataset_file) as f:
        dataset = json.load(f)
        for i in range(len(dataset)):
            dataset[i]["claim_id"] = i
        datapoints = [Datapoint.from_dict(d) for d in dataset]

    pipeline = Pipeline(
        retriever=MmrFaissRetriever(path=vecstore_path),
        evidence_generator=DynamicFewShotBatchedEvidenceGenerator(reference_corpus_path=train_file),
        classifier=NoTiebreakClassifier()
    )

    submission = []
    dump = []

    for dp in tqdm(datapoints[:1]):
        result = pipeline(dp)
        submission.append(result.to_submission())
        dump.append(result)

    pipeline.evidence_generator.get_batch_files(path=prompts, batch_size=9999)

    with open(f"{submission_path}/{PIPELINE_NAME}_unlabeled.json", "w") as f:
        json.dump(submission, f, indent=4)
    with open(f"{submission_path}/{PIPELINE_NAME}.pkl", "wb") as f:
        pickle.dump(dump, f)