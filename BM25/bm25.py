import pyterrier as pt
import pandas as pd
import os
import re

if not pt.started():
    pt.init()
    pt.ApplicationSetup.setProperty("querying.parser", "org.terrier.querying.parser.SingleTermQueryParser")

index_dir = os.path.abspath("pyterrier_index/train_bm25_positional")

if not os.path.exists(index_dir):
    raise FileNotFoundError(f"Could not find the index: {index_dir}")

index = pt.IndexFactory.of(index_dir)

bm25 = pt.terrier.Retriever(index, wmodel="BM25", metadata=["docno", "originalCode", "docstring", "code"])

tokenizer = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
def strip_markup(text):
    return " ".join(tokenizer.getTokens(text))

def search_code_snippet(query_snippet: str, language: str, top_k: int = 10) -> pd.DataFrame:
    """
    Returns the most similar documents using BM25 for a given code snippet.
    """
    query_snippet = re.sub(r'\n\s*\n', '\n', query_snippet)
    query_snippet = query_snippet.strip()
    query_snippet = strip_markup(query_snippet)

    if not query_snippet:
        raise ValueError("The query is empty after cleanup. Please provide a valid code snippet.")

    query_df = pd.DataFrame([["q1", query_snippet]], columns=["qid", "query"])

    results = bm25.transform(query_df)
    return results[["docno", "score", "originalCode", "docstring"]].head(top_k)