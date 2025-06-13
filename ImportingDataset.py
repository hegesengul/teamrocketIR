import gzip
import json
import os
from tqdm import tqdm

def extract_python_from_gz(files):
    data_python_dir = "data/python/final/jsonl/train"
    save_python_dir = "codesearchnet_cached/python"

    os.makedirs(save_python_dir, exist_ok=True) 
    output_file = os.path.join(save_python_dir, "train_data.jsonl")
    with open(output_file, "w", encoding="utf-8") as f_out:
        counter = 0
        for file_name in tqdm(files):
            gz_file_path = os.path.join(data_python_dir, file_name)
            with gzip.open(gz_file_path, 'rt', encoding='utf-8') as f_in:
                for line in f_in:
                    doc = json.loads(line)
                    if doc.get("language") == "python":
                        code = doc.get("code", "") 
                        docstring = doc.get("docstring", "") 
                        f_out.write(json.dumps({"docno": counter, "code": code, "docstring": docstring, "originalCode": code, "language": "python"}) + "\n")
                    counter = counter + 1
            
    print(f"[💾] Python code saved as {output_file}.")

def extract_java_from_gz(files):
    data_java_dir = "data/java/final/jsonl/train"
    save_java_dir = "codesearchnet_cached/java"

    os.makedirs(save_java_dir, exist_ok=True) 
    output_file = os.path.join(save_java_dir, "train_data.jsonl")

    with open(output_file, "w", encoding="utf-8") as f_out:
        counter = 0
        for file_name in tqdm(files):
            gz_file_path = os.path.join(data_java_dir, file_name)
            with gzip.open(gz_file_path, 'rt', encoding='utf-8') as f_in:
                for line in f_in:
                    doc = json.loads(line)
                    if doc.get("language") == "java":
                        code = doc.get("code", "") 
                        docstring = doc.get("docstring", "") 
                        f_out.write(json.dumps({"docno": counter, "code": code, "docstring": docstring, "originalCode": code, "language": "java"}) + "\n")
                    counter = counter + 1
            
    print(f"[💾] Java code saved as {output_file}.")

extract_python_from_gz([f"python_train_{i}.jsonl.gz" for i in range(14)])
extract_java_from_gz([f"java_train_{i}.jsonl.gz" for i in range(16)])
