import pyterrier as pt

pt.init()

dataset = pt.get_dataset('irds:codesearchnet')

for doc in dataset.get_corpus_iter():
    print(doc)