import pyterrier as pt
import torch



"""
from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
text = "def sqrt(x): return x**0.5"
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input,output_hidden_states=True)



text1 = "def sqrt(x): return x**0.5"

encoded_input1 = tokenizer(text1, return_tensors='pt')
output1 = model(**encoded_input1,output_hidden_states=True)
1==1

vec1=output.last_hidden_state[0][0]
vec2=output1.last_hidden_state[0][0]

res = (vec1 - vec2).pow(2).sum().sqrt()
res1 = torch.dot(vec1,vec2)
"""

def download_dataset():
    pt.init()
    dataset = pt.get_dataset('irds:codesearchnet')

    return list(map(lambda x: x["code"], dataset.get_corpus_iter()))

corpus = download_dataset()
#corpus = ["123", "adjsh"]

from transformers import RobertaTokenizer, RobertaModel
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

model = RobertaModel.from_pretrained('roberta-base')

encoded_input = tokenizer(corpus, return_tensors='pt', padding=True)
output = model(**encoded_input,output_haidden_states=True)

torch.save(output.last_hidden_state, "output.pt")