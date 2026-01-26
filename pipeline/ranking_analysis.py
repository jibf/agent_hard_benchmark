import argparse
import numpy as np
from scipy.stats import spearmanr, kendalltau

def count_indistinguishable(scores, tol=0.01):
    """
    Count the number of indistinguishable scores in a list.

    Two scores are considered indistinguishable if their difference is less than `tol`.
    A group of scores is considered indistinguishable if every score in the group
    is within `tol` of at least one other score in the group.

    Args:
        scores (list of float): List of scores.
        tol (float): Threshold for considering scores indistinguishable. Default is 0.01.

    Returns:
        int: Total number of scores that are indistinguishable with at least one other score.
    """
    if not scores:
        return 0

    scores = sorted(scores)
    n = len(scores)
    indist_count = 0
    visited = [False] * n

    for i in range(n):
        if visited[i]:
            continue
        group = [scores[i]]
        for j in range(i + 1, n):
            if abs(scores[j] - scores[i]) < tol:
                group.append(scores[j])
                visited[j] = True
        if len(group) > 1:
            indist_count += len(group)

    return indist_count

def analyze_ranking_changes_metric(old_rank_dict: dict, new_rank_dict: dict, k: int = 5):
    """
    Analyzes and summarizes the changes between two ranking lists.

    Args:
    old_ranking (list): A list of agent/item names in the old ranking.
    new_ranking (list): A list of agent/item names in the new ranking.
    k (int): The value of K for Top-K overlap calculation, defaults to 3.
    """
    old_ranking = sorted(old_rank_dict, key=lambda k: old_rank_dict[k])
    new_ranking = sorted(new_rank_dict, key=lambda k: new_rank_dict[k])
    
    # --- 1. Data Preparation ---
    # Ensure a consistent basis for comparison, including all agents from both lists
    all_items = sorted(list(set(old_ranking) | set(new_ranking)))
    
    if len(all_items) < 2:
        print("Error: Fewer than 2 items in the rankings, statistical analysis cannot be performed.")
        return

    # Create dictionaries mapping item names to their ranks
    old_rank_map = {item: rank + 1 for rank, item in enumerate(old_ranking)}
    new_rank_map = {item: rank + 1 for rank, item in enumerate(new_ranking)}

    # Assign a penalty rank to items that might appear in one list but not the other.
    # Here, we assume all items are in both lists. .get() would return None for missing
    # items, which would require more complex handling if lists can differ.
    penalty_rank = len(all_items) + 1
    old_ranks_numeric = [old_rank_map.get(item, penalty_rank) for item in all_items]
    new_ranks_numeric = [new_rank_map.get(item, penalty_rank) for item in all_items]

    # --- 4. Calculate Mean Absolute Rank Change ---
    mean_abs_change = np.mean(np.abs(np.array(old_ranks_numeric) - np.array(new_ranks_numeric)))
    print(f"   - Average Change: {mean_abs_change:.2f}")
    print("\n[3] Mean Absolute Rank Change")
    print(f"   - Interpretation: On average, each item's rank shifted by {mean_abs_change:.2f} positions. A higher value indicates a more drastic change.")

    # --- 5. Calculate Top-K Set Overlap ---
    if k > len(all_items):
        k = len(all_items)
        
    top_k_old = set(old_ranking[:k])
    top_k_new = set(new_ranking[:k])
    overlap_items = top_k_old.intersection(top_k_new)
    overlap_count = len(overlap_items)
    
    # print(f"\n[4] Top-{k} Set Overlap")
    # print(f"   - Old Top {k}: {old_ranking[:k]}")
    # print(f"   - New Top {k}: {new_ranking[:k]}")
    # print(f"   - Top{k} Overlap Count: {overlap_count} / {k}")
    # print(f"   - Overlapping Items: {list(overlap_items)}")
    # print(f"   - Interpretation: This metric reflects the stability of the top tier. Higher overlap means a more stable top tier.")

    """
    # --- 2. Calculate Spearman's Rank Correlation Coefficient (ρ) ---
    spearman_corr, spearman_p = spearmanr(old_ranks_numeric, new_ranks_numeric)
    print("\n[1] Spearman's Rank Correlation (Spearman's ρ)")
    print(f"   - Spearman's coefficient (ρ): {spearman_corr:.4f}")
    print(f"   - P-value: {spearman_p:.4f}")
    if spearman_p < 0.05:
        print("   - Interpretation: The result is statistically significant. The coefficient indicates the strength of the monotonic relationship between the two rankings.")
    else:
        print("   - Interpretation: The result is not statistically significant; the observed correlation may be due to chance.")

    # --- 3. Calculate Kendall's Tau Rank Correlation Coefficient (τ) ---
    kendall_corr, kendall_p = kendalltau(old_ranks_numeric, new_ranks_numeric)
    print("\n[2] Kendall's Tau Correlation (Kendall's τ)")
    print(f"   - Kendall's coefficient (τ): {kendall_corr:.4f}")
    print(f"   - P-value: {kendall_p:.4f}")
    if kendall_p < 0.05:
        print("   - Interpretation: The result is statistically significant. The coefficient measures the ordinal association between the two rankings.")
    else:
        print("   - Interpretation: The result is not statistically significant; the ordinal association may be due to chance.")
    
    
    print("\n--- End of Report ---")
    """

def analyze_model_rankings(file_path):
    # Read the CSV file

    def parse_csv_section(file_path, section_header):
        """Parse a specific section from the CSV file"""
        models = []
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Find the section
        start_idx = None
        for i, line in enumerate(lines):
            if section_header in line:
                start_idx = i + 1  # Skip the header line
                break

        if start_idx is None:
            return models

        # Parse models from the section
        for i in range(start_idx, len(lines)):
            line = lines[i].strip()
            if not line:
                break

            # Remove line number prefix (e.g., "19→")
            if '→' in line:
                line = line.split('→', 1)[1]

            # Split by comma and get model name and overall score
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 2 and not parts[0].startswith('Model') and parts[0] and parts[1]:
                model_name = parts[0].strip()
                try:
                    overall_score = float(parts[1])
                    models.append((model_name, overall_score))
                except ValueError:
                    continue

        # Sort by overall score (descending)
        models.sort(key=lambda x: x[1], reverse=True)
        return models

    # Parse each section
    baseline_models = parse_csv_section(file_path, "BASELINE - Model Performance")
    step1_models = parse_csv_section(file_path, "STEP1 - Model Performance")
    step2_models = parse_csv_section(file_path, "STEP2 - Model Performance")
    step4_models = parse_csv_section(file_path, "STEP4 - Model Performance")

    # Create ranking dictionaries
    def create_rankings(models):
        return {model[0]: idx + 1 for idx, model in enumerate(models)}

    baseline_rankings = create_rankings(baseline_models)
    step1_rankings = create_rankings(step1_models)
    step2_rankings = create_rankings(step2_models)
    step4_rankings = create_rankings(step4_models)

    # Analyze ranking changes
    def analyze_ranking_changes(baseline_ranks, step_ranks, step_name):
        changes = {}
        for model in baseline_ranks:
            if model in step_ranks:
                change = baseline_ranks[model] - step_ranks[model]  # Positive = improvement (lower rank number)
                changes[model] = change

        # Sort by absolute change magnitude
        improved = [(model, change) for model, change in changes.items() if change > 0]
        dropped = [(model, -change) for model, change in changes.items() if change < 0]

        improved.sort(key=lambda x: x[1], reverse=True)
        dropped.sort(key=lambda x: x[1], reverse=True)

        print(f"\n=== {step_name} vs BASELINE ===")

        if improved:
            print(f"\n IMPROVED Models:")
            for model, change in improved:
                baseline_rank = baseline_ranks[model]
                new_rank = step_ranks[model]
                print(f"  {model.split('/')[-1]}: #{baseline_rank} → #{new_rank} (+{change})")

        if dropped:
            print(f"\n DROPPED Models:")
            for model, change in dropped:
                baseline_rank = baseline_ranks[model]
                new_rank = step_ranks[model]
                print(f"  {model.split('/')[-1]}: #{baseline_rank} → #{new_rank} (-{change})")

        if not improved and not dropped:
            print("\n No ranking changes detected")
        
        print("Ranking Change Rate: {:.2f}%".format((len(improved) + len(dropped))/len(baseline_ranks) * 100))

    baseline_scores = [x[1] for x in baseline_models]
    step2_scores = [x[1] for x in step2_models]
    step4_scores = [x[1] for x in step4_models]
    baseline_num_indist = count_indistinguishable(baseline_scores)
    step2_num_indist = count_indistinguishable(step2_scores)
    step4_num_indist = count_indistinguishable(step4_scores)


    # Analyze each step
    analyze_ranking_changes(baseline_rankings, step1_rankings, "STEP1")
    analyze_ranking_changes_metric(baseline_rankings, step1_rankings)
    analyze_ranking_changes(baseline_rankings, step2_rankings, "STEP2")
    analyze_ranking_changes_metric(baseline_rankings, step2_rankings)
    analyze_ranking_changes(baseline_rankings, step4_rankings, "STEP4")
    analyze_ranking_changes_metric(baseline_rankings, step4_rankings)
    # print("baseline scores", baseline_scores)
    # print("step4 scores", step4_scores)
    print(f"num indistinguishable models: {baseline_num_indist} / {step2_num_indist} / {step4_num_indist}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze model ranking changes across different stages")
    parser.add_argument("--csv", required=True, help="Path to the CSV file containing model performance data")

    args = parser.parse_args()
    print(args.csv)
    analyze_model_rankings(args.csv)

    # test_score = [0.205, 0.308, 0.718, 0.710, 0.727, 0.737, 0.212, 0.505, 0.223, 0.225]
    # count_indistinguishable(test_score)