import pyterrier as pt
import pandas as pd
import os
import re
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

codeberta_tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
codeberta_model = AutoModel.from_pretrained("huggingface/CodeBERTa-small-v1")
codeberta_model.eval()

if not pt.started():
    pt.init()
    pt.ApplicationSetup.setProperty("querying.parser", "org.terrier.querying.parser.SingleTermQueryParser")

index_dir = os.path.abspath("pyterrier_index/train_bm25_positional")

if not os.path.exists(index_dir):
    raise FileNotFoundError(f"Could not find the index: {index_dir}")

index = pt.IndexFactory.of(index_dir)

with open("data/code_embeddings.pt", "rb") as f:
    precomputed_embeddings = torch.load(f)

bm25 = pt.terrier.Retriever(index, wmodel="BM25", metadata=["docno", "originalCode", "docstring", "code"])

tokenizer = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
def strip_markup(text):
    return " ".join(tokenizer.getTokens(text))


def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        # CLS token embedding (first token)
        return outputs.last_hidden_state[:, 0, :]  # shape: (1, hidden_size)
    
def search_code_snippet(query_snippet: str, language: str, top_k: int = 5) -> pd.DataFrame:
    """
    Returns the most similar documents using BM25 for a given code snippet.
    """
    query_snippet = re.sub(r'\n\s*\n', '\n', query_snippet)
    query_snippet = query_snippet.strip()
    query_snippet = strip_markup(query_snippet)

    if not query_snippet:
        raise ValueError("The query is empty after cleanup. Please provide a valid code snippet.")

    query_df = pd.DataFrame([["q1", query_snippet]], columns=["qid", "query"])

    initial_results = bm25.transform(query_df).head(50)

    # Step 2: Compute embeddings
    query_emb = get_embedding(query_snippet, codeberta_tokenizer, codeberta_model)  # shape: [1, hidden]
    codes = initial_results["docno"].tolist()
    doc_embeddings = torch.stack([precomputed_embeddings[docno] for docno in codes]) 

    # Step 3: Cosine similarity
    scores = F.cosine_similarity(query_emb, doc_embeddings)  # shape: [N]

    # Step 4: Add scores and rerank
    initial_results = initial_results.copy()
    initial_results["dense_score"] = scores.numpy()
    reranked = initial_results.sort_values("dense_score", ascending=False)

    return reranked[["docno", "dense_score", "originalCode", "docstring"]].head(top_k)