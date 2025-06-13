import pyterrier as pt
import pandas as pd
import os
import re
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import torch.nn.functional as F

torch.set_default_device("cuda")

codeberta_tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
codeberta_model = AutoModel.from_pretrained("huggingface/CodeBERTa-small-v1")
codeberta_model.eval()

roberta_tokenizer = AutoTokenizer.from_pretrained("roberta-base")
roberta_model = AutoModel.from_pretrained("roberta-base")
roberta_model.eval()

if not pt.started():
    pt.init()
    pt.ApplicationSetup.setProperty("querying.parser", "org.terrier.querying.parser.SingleTermQueryParser")

index_dir = os.path.abspath("pyterrier_index/train_bm25_positional")

if not os.path.exists(index_dir):
    raise FileNotFoundError(f"Could not find the index: {index_dir}")

index = pt.IndexFactory.of(index_dir)

bm25 = pt.terrier.Retriever(index, wmodel="BM25", metadata=["docno", "originalCode", "docstring", "code", "language"])

tokenizer = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
def strip_markup(text):
    return " ".join(tokenizer.getTokens(text))


def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
    with torch.no_grad():
        outputs = model(**inputs)
        # CLS token embedding (first token)
        return outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
    
def search_code_snippet(query_snippet: str, model: str, top_k: int = 5) -> pd.DataFrame:
    """
    Returns the most similar documents using BM25 for a given code snippet.
    """
    query_snippet = re.sub(r'\n\s*\n', '\n', query_snippet)
    query_snippet = query_snippet.strip()
    query_snippet = strip_markup(query_snippet)

    if not query_snippet:
        raise ValueError("The query is empty after cleanup. Please provide a valid code snippet.")
    if model == "BM25-CodeBERTa-small-v1":
        base_model = codeberta_model
        base_tokenizer = codeberta_tokenizer
    else:
        base_model = roberta_model
        base_tokenizer = roberta_tokenizer
    query_df = pd.DataFrame([["q1", query_snippet]], columns=["qid", "query"])

    initial_results = bm25.transform(query_df).head(50)

    # Step 2: Compute embeddings
    query_emb = get_embedding(query_snippet, base_tokenizer, base_model)  # shape: [1, hidden]
    codes = initial_results["docno"].tolist()
    # doc_embeddings = torch.stack([precomputed_embeddings[docno] for docno in codes]) 
    doc_embeddings = []
    codes = initial_results["originalCode"].tolist()

    for code in codes:
        emb = get_embedding(code, base_tokenizer, base_model) 
        doc_embeddings.append(emb)

    if not doc_embeddings:
        return []

    doc_embeddings = torch.cat(doc_embeddings, dim=0)  # shape: [N, hidden]
    differences = query_emb - doc_embeddings
    distances = torch.linalg.norm(differences, dim=-1) # shape: [N]
    initial_results = initial_results.copy()
    initial_results["euclidean_distance"] = distances.cpu().numpy()

    # Step 4: Add scores and rerank
    reranked = initial_results.sort_values("euclidean_distance", ascending=True)

    return reranked[["docno", "euclidean_distance", "originalCode", "docstring", "language"]].head(top_k).reset_index(drop=True)