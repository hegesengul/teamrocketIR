#%%
# Download corpus

import pyterrier as pt

def download_dataset():
    if not pt.java.started():
        pt.java.init()
    dataset = pt.get_dataset('irds:codesearchnet')

    return list(map(lambda x: x["code"], dataset.get_corpus_iter()))

corpus = download_dataset()
#%%
# Set up the tokenizer
from transformers import RobertaTokenizer, RobertaTokenizerFast

tokenizer = RobertaTokenizerFast.from_pretrained('roberta-base', use_fast=True)

def tokenizer_func(data):
    return tokenizer(data["code"], padding="max_length", truncation=True)

# Set up dataset and create tokens

from datasets import Dataset

dataset = Dataset.from_dict({"code": corpus})

tokenized_dataset = dataset.map(tokenizer_func, batched=True, num_proc=6)

tokenized_dataset.to_json("tokens.json")
#%%
# Set up model and load tokens
import torch
from torch.utils.data import DataLoader
from transformers import RobertaModel
from datasets import Dataset

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_default_device('cuda')


model = RobertaModel.from_pretrained('roberta-base')
model.eval()

tokenized_dataset = Dataset.from_json("tokens.json")

tokenized_dataset.set_format("torch", columns=["input_ids", "attention_mask"])
#%%
# Run model
import numpy as np

BATCH_SIZE = 300

data_loader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE)

results = []

with torch.no_grad():
    i=0
    for batch in data_loader:
        output = model(**batch, output_hidden_states=True)
        print("Progress:{} {:.5f}".format(i, i/len(data_loader)))
        #results.append(output.last_hidden_state.cpu())
        results.append(output.last_hidden_state.cpu().numpy())

        if len(results) == 10:
            #torch.save(torch.stack(results), f"outputs/output{i-9}-{i}.pt")
            np.savez_compressed(f"outputs/output{i-9}-{i}.npz", np.stack(results))
            results = []
        i+=1