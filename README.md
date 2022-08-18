# BERTserini


This repo is a release of our **BERTserini** model referenced in [End-to-End Open-Domain Question Answering with BERTserini](https://www.aclweb.org/anthology/N19-4013/). 


![Image of BERTserini](https://github.com/rsvp-ai/bertserini/blob/master/pipeline.png?raw=true)

We demonstrate an end-to-end Open-Domain question answering system that integrates BERT with the open-source [Pyserini](https://github.com/castorini/pyserini) information retrieval toolkit. Our system integrates best practices from IR with a BERT-based reader to identify answers from a large corpus of Wikipedia articles in an end-to-end fashion. We report significant improvements over previous results (such as [DrQA system](https://github.com/facebookresearch/DrQA)) on a standard benchmark test collection. It shows that fine-tuning pre-trained BERT with [SQuAD 1.1 Dataset](https://arxiv.org/abs/1606.05250) is sufficient to achieve high accuracy in identifying answer spans under Open Domain setting.

Following the Open Domain QA setting of DrQA, we are using Wikipedia as the large scale knowledge source of documents. The system first retrieves several candidate text segmentations among the entire knowledge source of documents, then read through the candidate text segments to determine the answers.

## Package Installation

```
conda create -n bertserini python==3.8.0
conda activate bertserini
conda install tqdm
pip install transformers==4.17
pip install pyserini==0.17.0
conda install -c pytorch faiss-gpu
pip install hanziconv
pip install zhon
pip install tensorboard
```

Also, install pytorch following instructions here: https://pytorch.org/get-started/locally/


## Development Installation
BERTserini requires Python 3.6+ and a couple Python dependencies. 
The repo is tested on Python 3.6, Cuda 10.1, PyTorch 1.5.1 on Tesla P40 GPUs.
Besides that, [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is recommended for convinence. Please run the following commands to install the Python dependencies. 
1. Clone the repo with ```git clone https://github.com/rsvp-ai/bertserini.git```
2. ```pip install -r requirements.txt -f --find-links https://download.pytorch.org/whl/torch_stable.html```

NOTE: Pyserini is the Python wrapper for Anserini. 
Please refer to their project [Pyserini](https://github.com/castorini/pyserini) for detailed usage. Also, Pyserini supports part of the features in Anserini; you can also refer to [Anserini](https://github.com/castorini/anserini) for more settings.


## A Simple Question-Answer Example
We provided an online interface to simply play with english QA [here](https://huggingface.co/rsvp-ai/bertserini-bert-base-squad?text=Where+do+I+live%3F&context=My+name+is+Sarah+and+I+live+in+London)

Below is a example for English Question-Answering. We also provide an example for Chinese Question-Answering [here](docs/qa_example_zh.md).
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
searcher = build_searcher("indexes/lucene-index.enwiki-20180701-paragraphs")
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

NOTE:

 The index we used above is English Wikipedia, which could be download via:
```
wget https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/pyserini-indexes/lucene-index.enwiki-20180701-paragraphs.tar.gz
```

After unzipping these file, we suggest you putting it in `indexes/`.

We have uploaded following finetuned checkpoints to the huggingace models:\
- [bertserini-bert-base-squad](https://huggingface.co/rsvp-ai/bertserini-bert-base-squad)
- [bertserini-bert-large-squad](https://huggingface.co/rsvp-ai/bertserini-bert-large-squad)

## Experiments
We have evaluated our system on `SQuAD 1.1` and `CMRC2018` development set.
Please see following documents for details:  
- [SQuAD experiments](docs/experiments-squad.md)  
- [CMRC experiments](docs/experiments-cmrc.md)

## Training
To finetune BERT on the SQuAD style dataset, please see [here](docs/train_squad.md) for details.


## DPR supporting

We enabled DPR retriever with [pyserini](https://github.com/castorini/pyserini) indexed corpus.
The corpus is created from the command:
```
python -m pyserini.encode \
    input   --corpus <original_corpus_dir> \
            --delimiter "DoNotApplyDelimiterPlease" \
            --shard-id 0 \
            --shard-num 1 \
    output  --embeddings dpr-ctx_encoder-multiset-base.<corpus_name> \
            --to-faiss \
    encoder --encoder facebook/dpr-ctx_encoder-multiset-base \
            --batch-size 16 \
            --device cuda:0 \
            --fp16  # if inference with autocast()
```

When enable dpr option in e2e inference, please set the following arguments:

```
--retriever dpr \
--encoder <path to dpr query encoder> \
--index_path <pyserini indexed dpr dir> \
--sparse_index <bm25 indexed corpus dir> \ # the dense index doesn't store the raw text, we need to get the original text from the sparse index
--device cuda:0
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
