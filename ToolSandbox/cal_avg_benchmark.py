import argparse
import json
import numpy as np


def calculate_average_benchmark(result_json):
    with open(result_json, "r") as f:
        results_dict = json.load(f)
    per_scenario_list = results_dict["per_scenario_results"]
    similarity_list = []
    for scrnario in per_scenario_list:
        similarity_list.append(scrnario["similarity"])
    
    print(f"The total similarity by scenario is {np.mean(np.array(similarity_list))} for {len(similarity_list)} scenarios")

    per_category_dict = results_dict["category_aggregated_results"]

    category_similarity_list = []
    for category in per_category_dict.values():
        category_similarity_list.append(category["similarity"])
    
    print(f"The total similarity by category is {np.mean(np.array(category_similarity_list))} for {len(category_similarity_list)} categories")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate average of values after colons in a json file')
    parser.add_argument('--log', type=str, required=True, help='Path to the json file')
    args = parser.parse_args()
    calculate_average_benchmark(args.log)