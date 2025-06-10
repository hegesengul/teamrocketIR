import json
import numpy as np
from sklearn.metrics import ndcg_score
from bm25 import search_code_snippet

def load_test_data(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    
def evaluate_ndcg(model_name: str, filepath: str, k: int = 5):
    test_queries = load_test_data(filepath)
    ndcg_scores = []
    counter = 0
    for item in test_queries:
        query = item["query"]
        relevant_snippets = set(s.strip() for s in item["relevant_docs"])

        results_df = search_code_snippet(query, model_name, top_k=k)

        y_true = []
        for code in results_df["originalCode"]:
            is_relevant = 1 if code.strip() in relevant_snippets else 0
            y_true.append(is_relevant)

        distances = results_df["euclidean_distance"].values
        scores = -distances 

        ndcg = ndcg_score([y_true], [scores], k=k)
        ndcg_scores.append(ndcg)
        print(f"Query: {counter}... → NDCG@{k}: {ndcg:.4f}")
        counter += 1

    avg_ndcg = np.mean(ndcg_scores)
    print(f"\n🔹 Average NDCG@{k}: {avg_ndcg:.4f}")

evaluate_ndcg("BM25-CodeBERTa-small-v1", "data/test_queries.json", k=5)
evaluate_ndcg("BM25-Roberta", "data/test_queries.json", k=5)
