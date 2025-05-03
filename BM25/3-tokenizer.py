import pyterrier as pt
import json
import os

if not pt.started():
    pt.init()

input_file = "codesearchnet_cached/train_python_preprocessed.jsonl"
index_dir = "C:/Users/fatih/OneDrive\Masaüstü/teamrocketIR/pyterrier_index/train_bm25_positional"
os.makedirs(os.path.dirname(index_dir), exist_ok=True)

corpus = []
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        corpus.append({
            "docno": obj["docno"],
            "text": obj["text"], 
            "code": obj["code"]   
        })

indexer = pt.IterDictIndexer(
    index_dir,
    meta={'docno': 64, 'text': 4096, 'code': 4096},
    text_attrs=['code'],
    blocks=True
)

indexref = indexer.index(corpus)
print(f"[💾] Index successfully saved at: {index_dir}")