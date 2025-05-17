import gzip
import json
import os
from tqdm import tqdm

def extract_python_from_gz(files):
    with open(output_file, "w", encoding="utf-8") as f_out:
        counter = 0
        for file_name in tqdm(files):
            gz_file_path = os.path.join(data_dir, file_name)
            with gzip.open(gz_file_path, 'rt', encoding='utf-8') as f_in:
                for line in f_in:
                    doc = json.loads(line)
                    if doc.get("language") == "python":
                        code = doc.get("code", "") 
                        docstring = doc.get("docstring", "") 
                        f_out.write(json.dumps({"docno": counter, "code": code, "docstring": docstring, "originalCode": code}) + "\n")
                    counter = counter + 1
            
    print(f"[💾] Python code saved as {output_file}.")

data_dir = "data/python/final/jsonl/train"
save_dir = "codesearchnet_cached/python"

os.makedirs(save_dir, exist_ok=True) 
output_file = os.path.join(save_dir, "train_data.jsonl")

if os.path.exists(output_file):
    print(f"[✓] File already exists: {output_file}. Skipping.")
else: 
    extract_python_from_gz([f"python_train_{i}.jsonl.gz" for i in range(14)])
