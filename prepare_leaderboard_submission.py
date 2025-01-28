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
        claim = sample['claim']
        label = sample['pred_label']
        prediction_evidence = ""
        for src_qa in sample['evidence']:
            prediction_evidence += src_qa["question"] + "\t\t\n" + src_qa["answer"] + "\t\t\n\n"
        #
        new_samples.append([i, claim, prediction_evidence, label, 'pred'])

    with open("leaderboard_submission/submission.csv".format(system_name), mode="w", newline="") as file:
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