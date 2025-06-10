import json

java_length = 454450
python_length = 412177
test_queries_java = []
with open("data/insert_test_queries_java.jsonl", "r", encoding="utf-8") as f:
    test_queries_java = [json.loads(line) for line in f]

with open("codesearchnet_cached/java/train_data.jsonl", "a", encoding="utf-8") as f:
    for i, doc in enumerate(test_queries_java):
        doc['docno'] = str(java_length + i)
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")

test_queries_python = []
with open("data/insert_test_queries_python.jsonl", "r", encoding="utf-8") as f:
    test_queries_python = [json.loads(line) for line in f]

with open("codesearchnet_cached/python/train_data.jsonl", "a", encoding="utf-8") as f:
    for i, doc in enumerate(test_queries_python):
        doc['docno'] = str(python_length + i)
        f.write(json.dumps(doc, ensure_ascii=False) + "\n")


