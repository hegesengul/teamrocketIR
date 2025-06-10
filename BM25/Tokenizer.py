import pyterrier as pt
import json
import os

if not pt.started():
    pt.init()

index_dir = os.path.abspath("pyterrier_index/train_bm25_positional")
python_input_file = "codesearchnet_cached/python/train_data_preprocessed.jsonl"
java_input_file = "codesearchnet_cached/java/train_data_preprocessed.jsonl"
os.makedirs(os.path.dirname(index_dir), exist_ok=True)

corpus = []
counter = 0
with open(python_input_file, "r", encoding="utf-8") as f:
    for line in f:
        counter += 1
        obj = json.loads(line)
        corpus.append({
            "docno": obj["docno"],
            "originalCode": obj["originalCode"], 
            "docstring": obj["docstring"], 
            "code": obj["code"],
            "language": obj["language"]   
        })
with open(java_input_file, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        corpus.append({
            "docno": str(int(obj["docno"]) + counter),
            "originalCode": obj["originalCode"], 
            "docstring": obj["docstring"], 
            "code": obj["code"],
            "language": obj["language"]   
        })

indexer = pt.IterDictIndexer(
    index_dir,
    meta={
        'docno': '64',
        'docstring': '4096',
        'code': '4096',
        'originalCode': '4096',
        'language': '64'
    },
    text_attrs=['code', 'docstring'],
    blocks=True
)

indexref = indexer.index(corpus)
print(f"[💾] Index successfully saved at: {index_dir}")