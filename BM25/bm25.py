import pyterrier as pt
import pandas as pd
import os
import re

if not pt.started():
    pt.init()
    pt.ApplicationSetup.setProperty("querying.parser", "org.terrier.querying.parser.SingleTermQueryParser")

index_dir = "C:/Users/fatih/OneDrive/Masaüstü/teamrocketIR/pyterrier_index/train_bm25_positional"

if not os.path.exists(index_dir):
    raise FileNotFoundError(f"Could not find the index: {index_dir}")

index = pt.IndexFactory.of(index_dir)

bm25 = pt.terrier.Retriever(index, wmodel="BM25", metadata=["docno", "text", "code"])

tokenizer = pt.autoclass("org.terrier.indexing.tokenisation.Tokeniser").getTokeniser()
def strip_markup(text):
    return " ".join(tokenizer.getTokens(text))

def search_code_snippet(query_snippet: str, top_k: int = 10) -> pd.DataFrame:
    """
    Returns the most similar documents using BM25 for a given code snippet.
    """
    query = re.sub(r'#.*', '', query_snippet)
    query = re.sub(r'""".*?"""', '', query, flags=re.DOTALL)
    query = re.sub(r"'''.*?'''", '', query, flags=re.DOTALL)
    query = query.replace("\n", " ").strip()
    query = query.replace("'", " ").strip()
    query = query.strip()
    query = strip_markup(query)

    if not query:
        raise ValueError("The query is empty after cleanup. Please provide a valid code snippet.")

    query_df = pd.DataFrame([["q1", query]], columns=["qid", "query"])

    results = bm25.transform(query_df)
    return results[["docno", "score", "text"]].head(top_k)