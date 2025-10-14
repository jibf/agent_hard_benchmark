import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity


def maximize_diversity_greedy(embeddings, target_size):
    n = embeddings.shape[0]
    if target_size >= n:
        return list(range(n))

    selected_indices = []
    remaining_indices = set(range(n))

    first_idx = np.random.choice(list(remaining_indices))
    selected_indices.append(first_idx)
    remaining_indices.remove(first_idx)

    sim_matrix = cosine_similarity(embeddings)

    while len(selected_indices) < target_size:
        min_dists = {}
        for idx in remaining_indices:
            dists = 1 - sim_matrix[idx, selected_indices]
            min_dists[idx] = np.min(dists)

        next_idx = max(min_dists, key=min_dists.get)
        selected_indices.append(next_idx)
        remaining_indices.remove(next_idx)

    return selected_indices

def compute_diversity(embeddings):

    # Cosine similarity matrix via dot product because embeddings are normalised
    sim_matrix = np.matmul(embeddings, embeddings.T)

    # Extract upper triangle (i < j)
    triu_indices = np.triu_indices(len(embeddings), k=1)
    cosine_sims = sim_matrix[triu_indices]
    avg_distance = np.mean(1 - cosine_sims)

    return float(avg_distance)

def main():
    embed_file = "tau-bench_embed_dict.pkl"
    with open(embed_file, "rb") as f:
        embeddings_dict = pickle.load(f)
    
    embeddings = np.array([embeddings_dict[qid] for qid in embeddings_dict])
    
    baseline_diversity = compute_diversity(embeddings)
    print(f"baseline diversity is {baseline_diversity} cross {len(embeddings)} questions")

    selected_indices = maximize_diversity_greedy(embeddings, target_size=int(len(embeddings)*0.6))

    selected_embeddings = embeddings[selected_indices]
    filtered_diversity = compute_diversity(selected_embeddings)

    print(f"filtered diversity is {filtered_diversity} cross {len(selected_embeddings)} questions")


if __name__ == "__main__":
    main()




    
