import json
import torch
from transformers import AutoTokenizer, AutoModel
import tqdm

codeberta_tokenizer = AutoTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
codeberta_model = AutoModel.from_pretrained("huggingface/CodeBERTa-small-v1")
codeberta_model.eval()

input_file = "codesearchnet_cached/python/train_data_preprocessed.jsonl"

def get_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.last_hidden_state[:, 0, :]  # CLS token embedding

embeddings = {}

with open(input_file, "r", encoding="utf-8") as f:
    for line in tqdm.tqdm(f):
        obj = json.loads(line)
        docno = obj["docno"]
        code = obj["originalCode"]

        if code and code.strip():
            emb = get_embedding(code, codeberta_tokenizer, codeberta_model)
            embeddings[docno] = emb.squeeze(0).cpu()

# Kaydet
torch.save(embeddings, "data/code_embeddings_CodeBERT.pt")
