import json
from collections import defaultdict
from utils.utils import *
from collections import Counter
import argparse


def basic_metric(result_dir):
    results = load_json(result_dir)
    domain_success = defaultdict(int)
    domain_turn_count = defaultdict(lambda: [0, 0])
    domain_call_count = defaultdict(lambda: [0, 0])
    complete_score_count = defaultdict(lambda: [0, 0])
    correct_score_count = defaultdict(lambda: [0, 0])
    for result in results:
        domain = result['id'].rsplit("-", 1)[0]
        if result['message'] == "Success.":
            domain_success[domain] += 1
        domain_turn_count[domain][0] += result['count_dict']['success_turn_num']
        domain_turn_count[domain][1] += result['count_dict']['total_turn_num']

        domain_call_count[domain][0] += result['count_dict']['correct_call_num']
        domain_call_count[domain][1] += result['count_dict']['total_call_num']

        if result["resp_eval"] is None:
            continue

        if result["resp_eval"]['complete']['score'] in {0, 1, 2}:
            complete_score_count[domain][0] += result["resp_eval"]['complete']['score']
            complete_score_count[domain][1] += 1
        
        if result["resp_eval"]['correct']['score'] in {0, 1, 2}:
            correct_score_count[domain][0] += result["resp_eval"]['correct']['score']
            correct_score_count[domain][1] += 1

    domain_success_rate = {k: v / 150 * 100 if k != "Cross" else v / 400 * 100 for k, v in domain_success.items()}
    domain_turn_acc = {k: v[0] / v[1] * 100 if v[1] != 0 else 0 for k, v in domain_turn_count.items()}
    domain_call_acc = {k: v[0] / v[1] * 100 if v[1] != 0 else 0 for k, v in domain_call_count.items()}

    overall_success = sum(domain_success.values()) / 1000 * 100
    overall_call_acc = sum([v[0] for v in domain_call_count.values()]) / sum([v[1] for v in domain_call_count.values()]) * 100

    complete_score, complete_total = 0, 0
    for k, v in complete_score_count.items():
        complete_score += v[0]
        complete_total += v[1]
    complete_score_avg = complete_score / complete_total if complete_total != 0 else 0

    correct_score, correct_total = 0, 0
    for k, v in correct_score_count.items():
        correct_score += v[0]
        correct_total += v[1]  
    correct_score_avg = correct_score / correct_total if correct_total != 0 else 0

    
    print(f"Domain Success Rate: {domain_success_rate}")
    # print(f"Domain Turn Accuracy: {domain_turn_acc}")
    print(f"Domain Call Accuracy: {domain_call_acc}")
    print(f"Overall Success Rate: {overall_success}")
    print(f"Overall Call Accuracy: {overall_call_acc}")
    print(f"Complete Score: {complete_score_avg}")
    print(f"Correct Score: {correct_score_avg}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="logs/test.log")
    parser.add_argument("--result_dir", type=str, default="result/../All.jsonl")
    args = parser.parse_args()
    basic_metric(args.result_dir)







if __name__ == "__main__":
    main()