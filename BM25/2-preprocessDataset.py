import json
import re
import os

def remove_comments_and_docstrings(code):
    code = re.sub(r'#.*', '', code)
    code = re.sub(r'""".*?"""', '', code, flags=re.DOTALL)
    code = re.sub(r"'''.*?'''", '', code, flags=re.DOTALL)
    
    return code

save_dir = "codesearchnet_cached"
input_file = os.path.join(save_dir, 'train_python.jsonl')
output_file = os.path.join(save_dir, 'train_python_preprocessed.jsonl')

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        doc = json.loads(line)
        preprocessed_code = remove_comments_and_docstrings(doc['text'])
        doc['code'] = preprocessed_code
        json.dump(doc, outfile)
        outfile.write('\n')

print(f"Preprocessed data has been saved to {output_file}")