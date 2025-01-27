#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zd302 at 12/01/2025
import csv
import json
import argparse

def convert(file_json, system_name):
    with open(file_json) as f:
        samples = json.load(f)

    new_samples = []
    for i, sample in enumerate(samples):
        if "gold" in file_json:
            claim = sample['claim']
            label = sample['label']
            reference_evidence = ""

            if "questions" in sample:
                for evidence in sample["questions"]:
                    # If the answers is not a list, make them a list:
                    if not isinstance(evidence["answers"], list):
                        evidence["answers"] = [evidence["answers"]]

                    for answer in evidence["answers"]:
                        reference_evidence += evidence["question"] + "\t\t\n" + answer["answer"]
                        #
                        if "answer_type" in answer and answer[
                            "answer_type"] == "Boolean" and "boolean_explanation" in answer:
                            reference_evidence += ". " + answer["boolean_explanation"]

                        reference_evidence += "\t\t\n\n"
            new_samples.append([i, claim, reference_evidence, label, 'gold'])

        if "pred" in file_json:
            claim = sample['claim']
            label = sample['pred_label']
            prediction_evidence = ""
            for src_qa in sample['evidence']:
                prediction_evidence += src_qa["question"] + "\t\t\n" + src_qa["answer"] + "\t\t\n\n"
            #
            new_samples.append([i, claim, prediction_evidence, label, 'pred'])

    if "gold" in file_json:
        with open("solution1.csv", mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "claim", "evi", "label", "split"])  # Write header
            writer.writerows(new_samples)

    if "pred" in file_json:
        with open("leaderboard_submission/{}_submission.csv".format(system_name), mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["id", "claim", "evi", "label", "split"])  # Write header
            writer.writerows(new_samples)

    print("{} have been converted to .csv".format(file_json))

def main():

    parser = argparse.ArgumentParser(description='Process annotation files')
    
    # Add arguments
    parser.add_argument('--filename', type=str, default='data_store/baseline/dev_veracity_prediction.json',
                        help='Dataset filename (default: dev)')
    parser.add_argument('--system_name', type=str, default='baseline',
                        help='Dataset filename (default: dev)')
    
    # Parse arguments
    args = parser.parse_args()

    convert(args.filename, args.system_name)

    print("Done.")


if __name__ == "__main__":
    main()