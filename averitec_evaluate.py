import pandas as pd
from huggingface_hub import hf_hub_download
import json
import numpy as np
import scipy
import nltk
from nltk import word_tokenize
import tqdm
import time
import argparse
import copy
# import properties
import google.generativeai as genai
import pandas as pd
import os
import sys
import importlib
import re
from openai import OpenAI




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


def pairwise_meteor(candidate, reference):
    return nltk.translate.meteor_score.single_meteor_score(word_tokenize(reference), word_tokenize(candidate))


def compute_all_pairwise_scores(src_data, tgt_data, metric):
    scores = np.empty((len(src_data), len(tgt_data)))

    for i, src in enumerate(src_data):
        for j, tgt in enumerate(tgt_data):
            scores[i][j] = metric(src, tgt)

    return scores


class AVeriTeCEvaluator:
    verdicts = [
        "Supported",
        "Refuted",
        "Not Enough Evidence",
        "Conflicting Evidence/Cherrypicking",
    ]
    pairwise_metric = None
    max_questions = 10
    metric = None
    averitec_reporting_levels = [0.25]

    def __init__(self, metric="meteor"):
        self.metric = metric
        if metric == "meteor":
            self.pairwise_metric = pairwise_meteor

    def evaluate_averitec_score(self, srcs, tgts):
        scores = []
        for i in tqdm.tqdm(range(len(srcs))):
            score = self.compute_pairwise_evidence_score(srcs.iloc[i], tgts.iloc[i])

            this_example_scores = [0.0 for _ in self.averitec_reporting_levels]
            for j, level in enumerate(self.averitec_reporting_levels):
                if score > level:
                    this_example_scores[j] = 1.0 if srcs.iloc[i]["label"] == tgts.iloc[i]["label"] else 0.0

            scores.append(this_example_scores)

        return np.mean(np.array(scores), axis=0), scores

    def evaluate_questions_only(self, srcs, tgts):
        all_utils = []

        for i in tqdm.tqdm(range(len(srcs))):
            src_questions, tgt_questions = [], []
            # prediction
            pred_evidence = srcs.iloc[i]['evi']
            pred_evi_pairs = pred_evidence.split('\t\t\n\n')

            for pred_qa in pred_evi_pairs:
                if pred_qa != '':
                    pred_question, pred_answer = pred_qa.split('\t\t\n')
                    src_questions.append(pred_question)

            src_questions = src_questions[: self.max_questions]

            # gold
            gold_evidence = tgts.iloc[i]['evi']
            gold_qa_pairs = gold_evidence.split('\t\t\n\n')

            for gold_qa in gold_qa_pairs:
                if gold_qa != '':
                    gold_question, gold_answer = gold_qa.split('\t\t\n')
                    if gold_question not in tgt_questions:
                        tgt_questions.append(gold_question)

            #
            pairwise_scores = compute_all_pairwise_scores(src_questions, tgt_questions, self.pairwise_metric)
            assignment = scipy.optimize.linear_sum_assignment(pairwise_scores, maximize=True)
            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

            # Reweight to account for unmatched target questions
            reweight_term = 1 / float(len(tgt_questions))
            assignment_utility *= reweight_term

            all_utils.append(assignment_utility)

        return np.mean(all_utils), all_utils

    def compute_pairwise_evidence_score(self, src, tgt):
        """Different key is used for reference_data and prediction.
        For the prediction, the format is
        {"evidence": [
            {
                "question": "What does the increased federal medical assistance percentage mean for you?",
                "answer": "Appendix A: Applicability of the Increased Federal Medical Assistance Percentage ",
                "url": "https://www.medicaid.gov/federal-policy-guidance/downloads/smd21003.pdf"
            }],
        "pred_label": "Supported"}
        And for the data with fold label:
        {"questions": [
            {
                "question": "Where was the claim first published",
                "answers": [
                    {
                        "answer": "It was first published on Sccopertino",
                        "answer_type": "Abstractive",
                        "source_url": "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/",
                        "source_medium": "Web text",
                        "cached_source_url": "https://web.archive.org/web/20201129141238/https://scoopertino.com/exposed-the-imac-disaster-that-almost-was/"
                    }
                ]
            }]
        "label": "Refuted"}
        """
        # prediction
        src_strings = []
        pred_evidence = src['evi']
        pred_qa_pairs = pred_evidence.split('\t\t\n\n')

        for qa_pair in pred_qa_pairs:
            if qa_pair != '':
                pred_question, pred_answer = qa_pair.split('\t\t\n')
                pred_qa_pairs = pred_question + " " + pred_answer
                src_strings.append(pred_qa_pairs)

        src_strings = src_strings[: self.max_questions]

        # gold
        tgt_strings = []
        gold_evidence = tgt['evi']
        gold_qa_pairs = gold_evidence.split('\t\t\n\n')

        for qa_pair in gold_qa_pairs:
            if qa_pair != '':
                gold_question, gold_answer = qa_pair.split('\t\t\n')
                gold_qa_pairs = gold_question + " " + gold_answer
                tgt_strings.append(gold_qa_pairs)

        #
        pairwise_scores = compute_all_pairwise_scores(src_strings, tgt_strings, self.pairwise_metric)
        assignment = scipy.optimize.linear_sum_assignment(pairwise_scores, maximize=True)
        assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

        # Reweight to account for unmatched target questions
        reweight_term = 1 / float(len(tgt_strings))
        assignment_utility *= reweight_term
        return assignment_utility

    def evaluate_questions_and_answers(self, srcs, tgts):
        all_utils = []

        for i in tqdm.tqdm(range(len(srcs))):
            # pred
            src_strings = []
            pred_evidence = srcs.iloc[i]['evi']
            pred_qa_pairs = pred_evidence.split('\t\t\n\n')

            for qa_pair in pred_qa_pairs:
                if qa_pair != '':
                    pred_question, pred_answer = qa_pair.split('\t\t\n')
                    pred_qa_pairs = pred_question + " " + pred_answer
                    src_strings.append(pred_qa_pairs)

            src_strings = src_strings[: self.max_questions]

            # gold
            tgt_strings = []
            gold_evidence = tgts.iloc[i]['evi']
            gold_qa_pairs = gold_evidence.split('\t\t\n\n')

            for qa_pair in gold_qa_pairs:
                if qa_pair != '':
                    gold_question, gold_answer = qa_pair.split('\t\t\n')
                    gold_qa_pair = gold_question + " " + gold_answer
                    tgt_strings.append(gold_qa_pair)

            pairwise_scores = compute_all_pairwise_scores(src_strings, tgt_strings, self.pairwise_metric)
            assignment = scipy.optimize.linear_sum_assignment(pairwise_scores, maximize=True)
            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()

            # Reweight to account for unmatched target questions
            reweight_term = 1 / float(len(tgt_strings))
            assignment_utility *= reweight_term

            all_utils.append(assignment_utility)

        return np.mean(all_utils), all_utils

    def extract_full_comparison_strings(self, example, is_target=True):
        example_strings = []

        if is_target:
            if "questions" in example:
                for evidence in example["questions"]:
                    # If the answers is not a list, make them a list:
                    if not isinstance(evidence["answers"], list):
                        evidence["answers"] = [evidence["answers"]]

                    for answer in evidence["answers"]:
                        example_strings.append(
                            evidence["question"] + " " + answer["answer"]
                        )
                        if (
                            "answer_type" in answer
                            and answer["answer_type"] == "Boolean" and "boolean_explanation" in answer
                        ):
                            example_strings[-1] += ". " + answer["boolean_explanation"]
                    if len(evidence["answers"]) == 0:
                        example_strings.append(
                            evidence["question"] + " No answer could be found."
                        )
        else:
            if "evidence" in example:
                for evidence in example["evidence"]:
                    example_strings.append(
                        evidence["question"] + " " + evidence["answer"]
                    )

        if "string_evidence" in example:
            for full_string_evidence in example["string_evidence"]:
                example_strings.append(full_string_evidence)
        return example_strings


class EV2REvaluator:

    verdicts = [
        "Supported",
        "Refuted",
        "Not Enough Evidence",
        "Conflicting Evidence/Cherrypicking",
    ]
    # Config
    MAX_RETRIES = 10
    ev2r_reporting_levels = [0.5]
    # LLM
    MAX_TOKENS = 3000
    TEMPERATURE = 0

    # -------------------------
    llamaapi_api_token = ""     # To obtain the LLAMA API token, please visit this URL: https://console.llmapi.com/en/dashboard
    llamaapi_client = OpenAI(api_key=llamaapi_api_token, base_url="https://api.llmapi.com/")
    # -------------------------

    def __init__(self, properties=None):
        self.properties = properties
        self.prompt_type = properties.PromptTypes("atomic_reference_prec_recall")
        self.prompt_type1 = properties.PromptTypes("atomic_question_reference_prec_recall")

    def prepare_dataset(self, srcs, tgts):
        pred_questions = []
        ref_questions = []
        pred_qa_pairs = []
        ref_qa_pairs = []

        for i in range(len(srcs)):
            # ------------------------- extract questions and QA pairs from src files
            src_qa_pairs = srcs.iloc[i]['evi']
            src_qa_pair_list = src_qa_pairs.split('\t\t\n\n')

            src_q_evidence = []
            for _qa_pair in src_qa_pair_list:
                _ques = _qa_pair.split('\t\t\n')[0]
                if _ques:
                    src_q_evidence.append(_ques)

            pred_questions.append(self.properties.AveritecEntry(claim=srcs.iloc[i]['claim'],
                                                       label=srcs.iloc[i]['label'],
                                                       evidence=" ".join(src_q_evidence),
                                                       id=srcs.iloc[i]['id']
                                                       ))
            pred_qa_pairs.append(self.properties.AveritecEntry(claim=srcs.iloc[i]['claim'],
                                                       label=srcs.iloc[i]['label'],
                                                       evidence=src_qa_pairs,
                                                       id=srcs.iloc[i]['id']
                                                       ))

            # ------------------------- extract questions and QA pairs from tgt files
            tgt_qa_pairs = tgts.iloc[i]['evi']
            tgt_qa_pair_list = tgt_qa_pairs.split('\t\t\n\n')

            tgt_q_evidence = []
            for _qa_pair in tgt_qa_pair_list:
                _ques = _qa_pair.split('\t\t\n')[0]
                if _ques:
                    tgt_q_evidence.append(_ques)

            ref_questions.append(self.properties.AveritecEntry(claim=tgts.iloc[i]['claim'],
                                                               label=tgts.iloc[i]['label'],
                                                               evidence=" ".join(tgt_q_evidence),
                                                               id=tgts.iloc[i]['id']
                                                               ))
            ref_qa_pairs.append(self.properties.AveritecEntry(claim=tgts.iloc[i]['claim'],
                                                              label=tgts.iloc[i]['label'],
                                                              evidence=tgt_qa_pairs,
                                                              id=tgts.iloc[i]['id']
                                                             ))

        return pred_questions, ref_questions, pred_qa_pairs, ref_qa_pairs

    def query_llama33_llamaapi(self, prompt):
        try:
            messages = [
                {"role": "user", "content": prompt},
            ]

            completion = self.llamaapi_client.chat.completions.create(
                messages=messages,
                model="llama3.3-70b",
                temperature=self.TEMPERATURE,
                max_tokens=self.MAX_TOKENS
            )
            response_llm = completion.choices[0].message.content
            matches = re.findall(r'\{(.*?)\}', response_llm, re.DOTALL)
            response = "{" + matches[0] + "}"
            return response

        except Exception as e:
            print(e)
            return ""


    def prepare_prompt(self, tgt_sample, pred_sample, input_type):
        """Formats prompt using dataset sample as input."""
        if input_type == "qa_pair":
            prompt = self.properties.PROMPT_MAPPING[self.prompt_type].format(tgt_sample.claim,
                                                                        tgt_sample.evidence,
                                                                        pred_sample.evidence)
        if input_type == "question":
            prompt = self.properties.PROMPT_MAPPING[self.prompt_type1].format(tgt_sample.claim,
                                                                        tgt_sample.evidence,
                                                                        pred_sample.evidence)
        return prompt

    def get_response_text(self, response):
        if type(response) == genai.types.generation_types.GenerateContentResponse:
            try:
                return response.text
            except Exception as e:
                print("Error in extracting Gemini response: {}".format(e))
                return ""
        else:
            return response

    def process_output(self, sample, response):
        logprob_inp = None
        return self.properties.OpenAIResponse(claim=sample.claim, evidence=sample.evidence,
                                         response=self.get_response_text(response),
                                         gold=sample.label.lower(), id=sample.id,
                                         logprobs=logprob_inp)

    def calculate_question_score_prec_recall_openai_response(self, response_llm):
        response_openai_copy = copy.deepcopy(response_llm)
        try:
            if type(response_llm.response) == str:
                response = json.loads(response_llm.response)
                # response = json.loads(
                #     response_llm.response.replace(": '", ": \"").replace("',", "\",").replace("':", "\":"))
            else:
                response = response_llm.response
            response_openai_copy.response = response
            response_openai_copy.response['precision'] = response["support predicted questions"] / response[
                "facts count predicted questions"]
            response_openai_copy.response['recall'] = response["support reference questions"] / response[
                "facts count reference questions"]
        except Exception as e:
            print("Following exception occurred: {}".format(e))
            return None

        return response_openai_copy

    def calculate_atomic_score_prec_recall_openai_response(self, response_llm):
        response_openai_copy = copy.deepcopy(response_llm)
        try:
            if type(response_llm.response) == str:
                response = json.loads(response_llm.response)
                # response = json.loads(
                #     response_llm.response.replace(": '", ": \"").replace("',", "\",").replace("':", "\":"))
            else:
                response = response_llm.response
            response_openai_copy.response = response
            response_openai_copy.response['precision'] = response["support predicted evidence"] / response[
                "facts count predicted evidence"]
            response_openai_copy.response['recall'] = response["support reference evidence"] / response[
                "facts count reference evidence"]
        except Exception as e:
            print("Following exception occurred: {}".format(e))
            return None

        return response_openai_copy

    def calculate_question_scores(self, responses):
        predictions_q_scores = []

        for i, res in enumerate(responses):
            pred_q_scores = self.calculate_question_score_prec_recall_openai_response(res)
            # if pred_w_scores:
            predictions_q_scores.append(pred_q_scores)

        return predictions_q_scores

    def calculate_prediction_scores(self, responses):
        predictions_w_scores = []

        for i, res in enumerate(responses):
            pred_w_scores = self.calculate_atomic_score_prec_recall_openai_response(res)
            # if pred_w_scores:
            predictions_w_scores.append(pred_w_scores)

        return predictions_w_scores

    def prompt_api_model(self, srcs, tgts, input_type):
        responses = []

        for i, tgt_sample in tqdm.tqdm(enumerate(tgts), desc="feed the prompt to api model ..."):
            print("{}/{}".format(i, len(tgts)))
            pred_sample = srcs[i]
            #
            prompt = self.prepare_prompt(tgt_sample, pred_sample, input_type)
            #
            attempt = 0
            while attempt < self.MAX_RETRIES:
                try:
                    response = self.query_llama33_llamaapi(prompt)
                    responses.append(self.process_output(tgt_sample, response))
                    print("One request successfully processed..")
                    break
                except:
                    attempt += 1
                    wait_time = 10 ** attempt  # Exponential backoff
                    print(f"Request timed out. Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)

        return responses

    def extract_ev2r_score(self, srcs, tgts, qa_evi_scores):
        scores = []
        ev2r_qa_recall = []

        for i in tqdm.tqdm(range(len(srcs))):
            this_example_scores = [0.0 for _ in self.ev2r_reporting_levels]
            #
            for k, ev2r_score in enumerate(qa_evi_scores):
                if ev2r_score and ev2r_score.id == i:
                    precision, recall = ev2r_score.response['precision'], ev2r_score.response['recall']
                    #
                    for j, level in enumerate(self.ev2r_reporting_levels):
                        if recall > level:
                            this_example_scores[j] = 1.0 if srcs.iloc[i]["label"] == tgts.iloc[i]["label"] else 0.0

                    scores.append(this_example_scores)
                    ev2r_qa_recall.append(recall)
                    break

                if ev2r_score and ev2r_score.id > i:
                    break

            if len(scores) != (i + 1):
                scores.append(this_example_scores)
                ev2r_qa_recall.append(0.0)

        return np.mean(np.array(scores), axis=0), scores, np.mean(np.array(ev2r_qa_recall), axis=0), ev2r_qa_recall

    def extract_recall_score(self, evi_scores):
        evi_recall = []

        for score in evi_scores:
            if score:
                precision, recall = score.response['precision'], score.response['recall']
                evi_recall.append(recall)
            else:
                evi_recall.append(0.0)

        return np.mean(np.array(evi_recall), axis=0), evi_recall


def compute(solution_file, submission_file):
    # import properties.py  (Huggingface competition)
    sys.path.append(os.path.dirname("properties.py"))
    properties = importlib.import_module("properties")

    # load golden and predicted file
    solution_df = pd.read_csv(solution_file)             # golden file
    submission_df = pd.read_csv(submission_file)         # predicted file

    # config on Huggingface competition
    public_ids = solution_df[solution_df.split == "gold"]['id'].values
    public_solution_df = solution_df[solution_df['id'].isin(public_ids)]
    public_submission_df = submission_df[submission_df['id'].isin(public_ids)]

    public_solution_df = public_solution_df.sort_values('id').reset_index(drop=True)
    public_submission_df = public_submission_df.sort_values('id').reset_index(drop=True)

    target_cols = [col for col in solution_df.columns if col not in ["split"]]

    # Evaluation on old AVeriTeC score (Hungarian meteor) and new AVeriTeC score (EV2R recall)
    # AVeriTeC Score
    scorer = AVeriTeCEvaluator()
    # Q only
    Q_evidence_score, Q_score_list = scorer.evaluate_questions_only(public_submission_df[target_cols], public_solution_df[target_cols])
    # Q + A
    QA_evidence_score, QA_score_list = scorer.evaluate_questions_and_answers(public_submission_df[target_cols], public_solution_df[target_cols])
    averitec_scores, averitec_score_list = scorer.evaluate_averitec_score(public_submission_df[target_cols], public_solution_df[target_cols])

    # EV2R Score
    EV2R_scorer = EV2REvaluator(properties)
    pred_questions, ref_questions, pred_qa_pairs, ref_qa_pairs = EV2R_scorer.prepare_dataset(public_submission_df[target_cols], public_solution_df[target_cols])
    # Q only
    q_responses = EV2R_scorer.prompt_api_model(pred_questions, ref_questions, input_type='question')
    q_evi_scores = EV2R_scorer.calculate_question_scores(q_responses)
    ev2r_q_recall, q_recall_list = EV2R_scorer.extract_recall_score(q_evi_scores)
    # Q + A
    qa_responses = EV2R_scorer.prompt_api_model(pred_qa_pairs, ref_qa_pairs, input_type='qa_pair')
    qa_evi_scores = EV2R_scorer.calculate_prediction_scores(qa_responses)
    ev2r_qa_scores, ev2r_qa_scores_list, ev2r_qa_recall, ev2r_qa_recall_list = EV2R_scorer.extract_ev2r_score(public_submission_df[target_cols], public_solution_df[target_cols], qa_evi_scores)
    #
    evaluation = {
        "public_score": {
            "Q only (Hungarian meteor)": "{}".format(round(Q_evidence_score, 4)),
            "Q + A (Hungarian meteor)": "{}".format(round(QA_evidence_score, 4)),
            "old AVeriTeC Score (Hungarian meteor)": "{}".format(round(averitec_scores[0], 4)),     # (meteor @ 0.25)
            "Q only (Ev2R recall)": "{}".format(round(ev2r_q_recall, 4)),
            "Q + A (Ev2R recall)": "{}".format(round(ev2r_qa_recall, 4)),
            "new AVeriTeC score (Ev2R recall)": "{}".format(round(ev2r_qa_scores[0], 4)),           # (recall @ 0.5)
        },
        "private_score": {
            "Q only (Hungarian meteor)": "{}".format(round(Q_evidence_score, 4)),
            "Q + A (Hungarian meteor)": "{}".format(round(QA_evidence_score, 4)),
            "old AVeriTeC Score (Hungarian meteor)": "{}".format(round(averitec_scores[0], 4)),     # (meteor @ 0.25)
            "Q only (Ev2R recall)": "{}".format(round(ev2r_q_recall, 4)),
            "Q + A (Ev2R recall)": "{}".format(round(ev2r_qa_recall, 4)),
            "new AVeriTeC score (Ev2R recall)": "{}".format(round(ev2r_qa_scores[0], 4)),           # (recall @ 0.5)
        }
    }
    print("\n*****Results of Submission *****\n")
    print(evaluation)

    return evaluation


def main():
    parser = argparse.ArgumentParser(description='Process annotation files')
    # Add arguments
    # convert golden_dev.json to solution.csv  (https://github.com/Raldir/FEVER-8-Shared-Task/blob/main/prepare_leaderboard_submission.py)
    # convert prediction_dev.json to submission.csv
    parser.add_argument('--label_file', type=str, default='evaluation/solution.csv',
                        help='Golden data filename.')
    parser.add_argument('--prediction_file', type=str, default='leaderboard_submission/submission.csv',
                        help='Predicted data filename')
    # Parse arguments
    args = parser.parse_args()


    download_nltk_data('punkt')
    download_nltk_data('punkt_tab')
    download_nltk_data('wordnet')
    
    #
    compute(args.label_file, args.prediction_file)


if __name__ == "__main__":
    main()