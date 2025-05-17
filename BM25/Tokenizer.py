import pyterrier as pt
import json
import os

if not pt.started():
    pt.init()

input_file = "codesearchnet_cached/python/train_data_preprocessed.jsonl"
index_dir = os.path.abspath("pyterrier_index/train_bm25_positional")
os.makedirs(os.path.dirname(index_dir), exist_ok=True)

corpus = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        corpus.append({
            "docno": obj["docno"],
            "originalCode": obj["originalCode"], 
            "docstring": obj["docstring"], 
            "code": obj["code"]   
        })

indexer = pt.IterDictIndexer(
    index_dir,
    meta={
        'docno': '64',
        'docstring': '4096',
        'code': '4096',
        'originalCode': '4096'
    },
    text_attrs=['code', 'docstring'],
    blocks=True
)

indexref = indexer.index(corpus)
print(f"[💾] Index successfully saved at: {index_dir}")