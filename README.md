# BERTserini


This is a release of our **BERTserini** model referenced in [End-to-End Open-Domain Question Answering with BERTserini](https://www.aclweb.org/anthology/N19-4013/). 


![Image of BERTserini](https://github.com/rsvp-ai/bertserini/blob/master/pipeline.png?raw=true)

We demonstrate an end-to-end Open-Domain question answering system that integrates BERT with the open-source [Pyserini](https://github.com/castorini/pyserini) information retrieval toolkit. Our system integrates best practices from IR with a BERT-based reader to identify answers from a large corpus of Wikipedia articles in an end-to-end fashion. We report large improvements over previous results (such as [DrQA system](https://github.com/facebookresearch/DrQA)) on a standard benchmark test collection. It is showing that fine-tuning pretrained BERT with [SQuAD 1.1 Dataset](https://arxiv.org/abs/1606.05250) is sufficient to achieve high accuracy in identifying answer spans under Open Domain setting.

Following the Open Domain QA setting of DrQA, we are using Wikipedia as the large scale knowledge source of documents. In order to anwer questions, the system first retrieve several candidate text segmentations among the entire knowledge source of documents, then read through the candidate text segments to determine the answers.

# Quick Start

1. [Install dependencies](#install-dependencies)
2. [Download the PreBuilt Wikipedia index](#download-prebuilt-wikipedia-index)
3. Download the pretrained models
4. 



We have uploaded the finetuned checkpoints to the huggingface models: \
[bertserini-bert-base-squad](https://huggingface.co/rsvp-ai/bertserini-bert-base-squad) \
[bertserini-bert-large-squad](https://huggingface.co/rsvp-ai/bertserini-bert-large-squad) 

We will also provide the Chinese version of this pipeline on [CMRC](https://github.com/ymcui/cmrc2018) and [DRCD](https://github.com/DRCKnowledgeTeam/DRCD) datasets. \
% TODO: Chinese version is not ready yet.

# How to Run

Next, we describe how to run the pipeline.

## Install dependencies
First step is to prepare the python dependencies. \
Pyserini is the repo that wrap Anserini with python APIs. 
Please refer to their repo [Pyserini](https://github.com/castorini/pyserini) for detailed useage. Also, this wrapper only contain some of the features in Anserini, you can also refer to [Anserini](https://github.com/castorini/anserini) for more settings.

```
conda create -n bertserini
conda activate bertserini
conda install tqdm
pip install pyserini
pip install transformers 
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html # or install torch according to your cuda version
pip install tensorboardX
```

## Download PreBuilt Wikipedia Index

We have indexed the 20180701 wikipedia dump used in DrQA with Anserini, you can download the prepared index here:
```
wget ftp://72.143.107.253/BERTserini/index.zip
````
index.zip contains the indexed latest wikipedia dump with Answerini.
After unzip these files, put them under the root path of this repo and then you are ready to go.
Take the folloing folder structure as example:
```
bertserini
+--- index
|    +--- lucene-index.enwiki-20180701-paragraphs
|    |    +--- ...
+--- otherfiles under this repo
```

To get the index on your own corpus, please refer to [Pyserini](https://github.com/castorini/pyserini).

## Download datasets:

```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

## Run inferencing SQuAD under open-domain setting

Set the ckpt information in the below script, according to the path after training. \
We have upload the checkpoints to hugging face, you can directly use them with the argments```--model_name_or_path rsvp-ai/bertserini-bert-base-squad``` or ```--model_name_or_path rsvp-ai/bertserini-bert-large-squad``` in the ```inference.sh``` script. \ 
Then run:
```
bash inference.sh
```
It will generate inference results on SQuAD, under the path of ./results/

## evaluation
Set the result path according to the inference result path
```
bash eval.sh
```
This will first automatically select the parameter to aggregate paragraph score (from Pyserini) and phrase score (from BERT), and finally select the best parameter and print the evaluation matrixs.
```
# expected result:

## BERT-large-wwm-uncased
(0.4, {'exact_match': 40.89877010406812, 'f1': 48.827808932780215, 'recall': 50.644587225343955, 
'precision': 49.308238592369754, 'cover': 46.87795648060549, 'overlap': 57.28476821192053})

## BERT-base-uncased
(0.5, {'exact_match': 39.89593188268685, 'f1': 47.58710784120026, 'recall': 49.27586877280707, 'precision': 48.10849111109448, 'cover': 45.31693472090823, 'overlap': 56.00756859035005})
```

## Run training
You can also train the base model from other pretrained model as long as it is already supporting Question Answering. \

Please set the correct parameters in the following script and then run.
```
bash train.sh
```
This script will run the training for BERT on SQuAD dataset.
It will generate checkpoints under ./tmp/


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
