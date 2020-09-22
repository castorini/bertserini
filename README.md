# BERTserini


This repo is a release of our **BERTserini** model referenced in [End-to-End Open-Domain Question Answering with BERTserini](https://www.aclweb.org/anthology/N19-4013/). 


![Image of BERTserini](https://github.com/rsvp-ai/bertserini/blob/master/pipeline.png?raw=true)

We demonstrate an end-to-end Open-Domain question answering system that integrates BERT with the open-source [Pyserini](https://github.com/castorini/pyserini) information retrieval toolkit. Our system integrates best practices from IR with a BERT-based reader to identify answers from a large corpus of Wikipedia articles in an end-to-end fashion. We report significant improvements over previous results (such as [DrQA system](https://github.com/facebookresearch/DrQA)) on a standard benchmark test collection. It shows that fine-tuning pre-trained BERT with [SQuAD 1.1 Dataset](https://arxiv.org/abs/1606.05250) is sufficient to achieve high accuracy in identifying answer spans under Open Domain setting.

Following the Open Domain QA setting of DrQA, we are using Wikipedia as the large scale knowledge source of documents. The system first retrieves several candidate text segmentations among the entire knowledge source of documents, then read through the candidate text segments to determine the answers.

# Quick Start

1. [Install dependencies](#package-installation)
2. [Download the PreBuilt Wikipedia index](#download-prebuilt-wikipedia-index)
3. [Download the pretrained models](#download-the-pretrained-models)
4. [Quickly start the Demo](#start-the-demo)


## Package Installation
```
pip install bertserini
```

## Development Installation
BERTserini requires Python 3.6+ and a couple Python dependencies. 
The repo is tested on Python 3.6, Cuda 10.1, PyTorch 1.5.1 on Tesla P40 GPUs.
Besides that, [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is recommended for convinence. Please run the following commands to install the Python dependencies. 
1. Clone the repo with ```git clone https://github.com/rsvp-ai/bertserini.git```
2. ```pip install -r requirements.txt```

NOTE: Pyserini is the Python wrapper for Anserini. 
Please refer to their project [Pyserini](https://github.com/castorini/pyserini) for detailed usage. Also, Pyserini supports part of the features in Anserini; you can also refer to [Anserini](https://github.com/castorini/anserini) for more settings.


## A Simple Question-Answer Example
```python
from bertserini.reader.base import Question, Context
from bertserini.reader.bert_reader import BERT
from bertserini.utils.utils_new import get_best_answer

model_name = "rsvp-ai/bertserini-bert-base-squad"
tokenizer_name = "rsvp-ai/bertserini-bert-base-squad"
bert_reader = BERT(model_name, tokenizer_name)

# Here is our question:
question = Question("Why did Mark Twain call the 19th century the glied age?")

# Option 1: fetch some contexts from Wikipedia with Pyserini
from bertserini.retriever.pyserini_retriever import retriever, build_searcher
searcher = build_searcher("/path/to/enwiki/index/")
contexts = retriever(question, searcher, 10)

# Option 2: hard-coded contexts
contexts = [Context('The "Gilded Age" was a term that Mark Twain used to describe the period of the late 19th century when there had been a dramatic expansion of American wealth and prosperity.')]

# Either option, we can ten get the answer candidates by reader
# and then select out the best answer based on the linear 
# combination of context score and phase score
candidates = bert_reader.predict(question, contexts)
answer = get_best_answer(candidates, 0.45)
print(answer.text)

```

## Citation

Please cite [the NAACL 2019 paper]((https://www.aclweb.org/anthology/N19-4013/)):

```
@article{yang2019end,
  title={End-to-end open-domain question answering with bertserini},
  author={Yang, Wei and Xie, Yuqing and Lin, Aileen and Li, Xingyu and Tan, Luchen and Xiong, Kun and Li, Ming and Lin, Jimmy},
  journal={arXiv preprint arXiv:1902.01718},
  year={2019}
}
```
