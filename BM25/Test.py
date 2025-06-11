import json
import numpy as np
from sklearn.metrics import ndcg_score
from bm25 import search_code_snippet
import matplotlib.pyplot as plt

def load_test_data(filepath: str):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    
def evaluate_ndcg_recall(model_name: str, filepath: str, k: int = 5):
    test_queries = load_test_data(filepath)
    ndcg_scores = []
    recall_scores = []
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

        if len(relevant_snippets) > 0:
            recall = sum(y_true) / len(relevant_snippets)
        else:
            recall = 0.0
        recall_scores.append(recall)

    avg_ndcg = np.mean(ndcg_scores)
    avg_recall = np.mean(recall_scores)
    return avg_ndcg, avg_recall

models = ["BM25-CodeBERTa-small-v1", "BM25-Roberta"]

ndcg_results = []
recall_results = []

for model in models:
    avg_ndcg, avg_recall = evaluate_ndcg_recall(model, "data/test_queries.json", k=5)
    ndcg_results.append(avg_ndcg)
    recall_results.append(avg_recall)
    print(f"{model}: Avg NDCG@5 = {avg_ndcg:.4f}, Avg Recall@5 = {avg_recall:.4f}")

x = np.arange(len(models))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))
bars1 = ax.bar(x - width/2, ndcg_results, width, label='NDCG@5')
bars2 = ax.bar(x + width/2, recall_results, width, label='Precision@5')

ax.set_ylabel('Score')
ax.set_title('Model Comparison on NDCG@5 and Recall@5')
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0, 1)
ax.legend()

def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom')

autolabel(bars1)
autolabel(bars2)

plt.tight_layout()
plt.savefig("ndcg_recall_comparison.png", dpi=300)
plt.close()