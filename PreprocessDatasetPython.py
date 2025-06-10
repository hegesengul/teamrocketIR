import json
import os
from nltk.tokenize import word_tokenize
import re

# stop_words = set(stopwords.words('english'))

def remove_comments_and_docstrings_with_regex(source_code):
    source_code = re.sub(r'("""|\'\'\')(.*?)(\1)', '', source_code, flags=re.DOTALL)
    source_code = re.sub(r'#.*', '', source_code)
    source_code = re.sub(r'\n\s*\n', '\n', source_code)
    return source_code
    
def clean_docstring(docstring):
    docstring = docstring.lower()
    docstring = docstring.replace('\n', ' ')
    tokens = word_tokenize(docstring)
    # filtered_tokens = [word for word in tokens if word not in stop_words and word.isalpha()] use inverted index
    return ' '.join(tokens)

save_dir = "codesearchnet_cached/python"
input_file = os.path.join(save_dir, 'train_data.jsonl')
output_file = os.path.join(save_dir, 'train_data_preprocessed.jsonl')

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        doc = json.loads(line)
        source_code = doc['code']
        preprocessed_code = remove_comments_and_docstrings_with_regex(source_code)
        if 'docstring' in doc and doc['docstring']:
            doc['docstring'] = clean_docstring(doc['docstring'])
        else:
             doc['docstring'] = ""
        doc["docno"] = str(doc["docno"])
        doc["code"] = preprocessed_code
        json.dump(doc, outfile, ensure_ascii=False)
        outfile.write('\n')

print(f"Preprocessed data has been saved to {output_file}")
