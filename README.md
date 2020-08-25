# BERTserini


This repo is a release of our **BERTserini** model referenced in [End-to-End Open-Domain Question Answering with BERTserini](https://www.aclweb.org/anthology/N19-4013/). 


![Image of BERTserini](https://github.com/rsvp-ai/bertserini/blob/master/pipeline.png?raw=true)

We demonstrate an end-to-end Open-Domain question answering system that integrates BERT with the open-source [Pyserini](https://github.com/castorini/pyserini) information retrieval toolkit. Our system integrates best practices from IR with a BERT-based reader to identify answers from a large corpus of Wikipedia articles in an end-to-end fashion. We report significant improvements over previous results (such as [DrQA system](https://github.com/facebookresearch/DrQA)) on a standard benchmark test collection. It shows that fine-tuning pre-trained BERT with [SQuAD 1.1 Dataset](https://arxiv.org/abs/1606.05250) is sufficient to achieve high accuracy in identifying answer spans under Open Domain setting.

Following the Open Domain QA setting of DrQA, we are using Wikipedia as the large scale knowledge source of documents. The system first retrieves several candidate text segmentations among the entire knowledge source of documents, then read through the candidate text segments to determine the answers.

# Quick Start

1. [Install dependencies](#install-dependencies)
2. [Download the PreBuilt Wikipedia index](#download-prebuilt-wikipedia-index)
3. [Download the pretrained models](#download-the-pretrained-models)
4. [Quickly start the Demo](#start-the-demo)


## Install dependencies

BERTserini requires Python 3.5+ and a couple Python dependencies. The repo is tested on Python 3.6, Cuda 10.1, PyTorch 1.5.1 on Tesla P40 GPUs.
Besides that, [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) is recommended for convinence. Please run the following commands to install the Python dependencies. 

```
conda create -n bertserini
conda activate bertserini
conda install tqdm
pip install pyserini
pip install transformers 
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html # or install torch according to your cuda version
pip install tensorboardX
pip install hanziconv # for chinese processing
```

NOTE: Pyserini is the Python wrapper for Anserini. 
Please refer to their project [Pyserini](https://github.com/castorini/pyserini) for detailed usage. Also, Pyserini supports part of the features in Anserini; you can also refer to [Anserini](https://github.com/castorini/anserini) for more settings.


## Download PreBuilt Wikipedia Index

We have indexed the 20180701 Wikipedia dump used in DrQA with Anserini; you can download the prepared index here:
```
wget ftp://72.143.107.253/BERTserini/english_wiki_2018_index.zip
````
For the chinese index, please download through:
```
wget ftp://72.143.107.253/BERTserini/chinese_wiki_2018_index.zip
```
```*index.zip``` contains the indexed latest Wikipedia dump with Anserini.

After unzipping these files, put them under the root path of this repo, and then you are ready to go.
Take the following folder structure as an example:
```
bertserini
+--- index
|    +--- lucene-index.enwiki-20180701-paragraphs
|    |    +--- ...
+--- other files under this repo
```

## Download the pre-trained models

We have uploaded the finetuned checkpoints to the huggingface models: \
[bertserini-bert-base-squad](https://huggingface.co/rsvp-ai/bertserini-bert-base-squad) \
[bertserini-bert-large-squad](https://huggingface.co/rsvp-ai/bertserini-bert-large-squad) \
[bertserini-bert-base-cmrc](https://huggingface.co/rsvp-ai/bertserini-bert-base-cmrc) # this is for Chinese \
[bertserini-roberta-base](https://huggingface.co/rsvp-ai/bertserini-roberta-base)

To run our finetuned model, just set ```--model_name_or_path rsvp-ai/<MODEL_NAME>``` for example: ```--model_name_or_path rsvp-ai/bertserini-bert-large-squad```.

We also provide the Chinese version of this pipeline on [CMRC](https://github.com/ymcui/cmrc2018) and [DRCD](https://github.com/DRCKnowledgeTeam/DRCD) datasets. 

## Start the Demo

To quickly try out the system, you should follow ```demo.sh``` to set the paths, then
```
bash demo.sh
``` 
or
```
bash demo_cmrc.sh
```
You can try our fine-tuned model with the Wikipedia articles.

# Training, inferencing and evaluating using your data

You may use your data on this system; we provide the steps based on the SQuAD dataset.

## Prepare index files:
To get the index on your corpus, please refer to [Pyserini](https://github.com/castorini/pyserini#how-do-i-search-my-own-documents). 

After getting the index, put it under the path ```bertserini/index/```

## Prepare the training datasets:

```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

For CMRC dataset, please refer to [CMRC](https://github.com/ymcui/cmrc2018).
TODO: for DRCD dataset, which is in Traditional Chinese, we have implemented the to-simplified argment, however, the results has not been tested.

## Training
Please set the correct parameters in the following script and then run.
```
bash train.sh
```
or, for chinese:
```
bash train_cmrc.sh
```

This script will run the training for BERT on the SQuAD dataset.
It will generate checkpoints under ./tmp/ \
You can also train the base model from other pre-trained models as long as it is already supporting Question Answering. 

## Inferencing SQuAD under the open-domain setting

Set the checkpoint information in the below script, according to the path after training. \
We have upload the finetuned checkpoints to hugging face, you can directly use them with the argments```--model_name_or_path rsvp-ai/bertserini-bert-base-squad``` or ```--model_name_or_path rsvp-ai/bertserini-bert-large-squad``` in the ```inference.sh``` script. \ 
Then run:
```
bash inference.sh
```
or, for chinese:
```
inference_cmrc.sh #this added --chinese argments to switch to chinese example processing
```
It will generate inference results on SQuAD, under the path of ./results/

## Evaluation
Set the result paths according to the inference result path
```
bash eval.sh
```
This script will first automatically select the parameter to aggregate paragraph score (from Pyserini) and phrase score (from BERT), and finally, select the best parameter and print the evaluation matrixs.
```
# expected results:

## rsvp-ai/bertserini-large-squad, this is finetuned based on bert-large-wwm-uncased
(0.4, {'exact_match': 41.54210028382214, 'f1': 49.45378799697662, 'recall': 51.119838584003105, 'precision': 49.8395951713666, 'cover': 47.228003784295176, 'overlap': 57.6631977294229})

## rsvp-ai/bertserini-bert-base-squad, this is finetuned based on bert-base-uncased
(0.5, {'exact_match': 39.89593188268685, 'f1': 47.58710784120026, 'recall': 49.27586877280707, 'precision': 48.10849111109448, 'cover': 45.31693472090823, 'overlap': 56.00756859035005})

## rsvp-ai/bertserini-bert-base-cmrc, this is bert-base-chinese finetuned on the chinese reading comprehension dataset(CMRC)
(0.5, {'f1_score': 68.0033167812909, 'exact_match': 51.164958061509786, 'total_count': 3219, 'skip_count': 1})
```

## Notes

We also provide the code to run with Anseirni's indexing version. \
This requires .jar files from compiled [Anserini](https://github.com/castorini/anserini). \
You can look into Anserini's repo and modify the code for you own needs. \
And then swithch to the API connecting Anserini provided in ./retriever/anserini_retriever.py #TODO: swithch to argument setting


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
