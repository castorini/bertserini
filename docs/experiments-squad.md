# Bertserini: Baseline on SQUAD QA

1. Clone the repo with ```git clone https://github.com/castorini/bertserini.git```
2. ```pip install -r requirements.txt -f --find-links https://download.pytorch.org/whl/torch_stable.html```

## Download PreBuilt Wikipedia Index

We have indexed the 20180701 Wikipedia dump used in DrQA with Anserini; you can download the prepared index here:
```
cd indexes
wget https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/pyserini-indexes/lucene-index.enwiki-20180701-paragraphs.tar.gz
tar -xvf lucene-index.enwiki-20180701-paragraphs.tar.gz
rm lucene-index.enwiki-20180701-paragraphs.tar.gz
cd ..
```
It contains the indexed 20180701 Wikipedia dump with Anserini.

## Download the pre-trained models

We have uploaded the finetuned checkpoints to the huggingface models: \
[bertserini-bert-base-squad](https://huggingface.co/rsvp-ai/bertserini-bert-base-squad) \
[bertserini-bert-large-squad](https://huggingface.co/rsvp-ai/bertserini-bert-large-squad) \
[bertserini-roberta-base](https://huggingface.co/rsvp-ai/bertserini-roberta-base)

To run our finetuned model, just set ```--model_name_or_path rsvp-ai/<MODEL_NAME>```   
For example: ```--model_name_or_path rsvp-ai/bertserini-bert-large-squad```.

# Inferencing and evaluating

## Prepare the datasets:

```
cd data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
cd ..
```

## Inferencing SQuAD under the open-domain setting
For `rsvp-ai/bertserini-bert-base-squad`
```
python -m bertserini.experiments.inference --dataset_path data/dev-v1.1.json \
                                           --index_path indexes/lucene-index.enwiki-20180701-paragraphs \
                                           --model_name_or_path rsvp-ai/bertserini-bert-base-squad \
                                           --output prediction/squad_bert_base_pred.json \
                                           --topk 10

```

For `rsvp-ai/bertserini-bert-large-squad`
```
python -m bertserini.experiments.inference --dataset_path data/dev-v1.1.json \
                                           --index_path indexes/lucene-index.enwiki-20180701-paragraphs \
                                           --model_name_or_path rsvp-ai/bertserini-bert-large-squad \
                                           --output prediction/squad_bert_large_pred.json \
                                           --topk 10

```
## Evaluation

```
mkdir temp
python -m bertserini.experiments.evaluate --eval_data data/dev-v1.1.json \
                                          --search_file prediction/squad_bert_<base/large>_pred.json \
                                          --output_path temp \
                                          --dataset squad
                                          
```
Expected results:
```
## rsvp-ai/bertserini-large-squad, this is finetuned based on bert-large-wwm-uncased
(0.4, {'exact_match': 41.81646168401135, 'f1': 49.697937151721774, 'recall': 51.37331878403011, 'precision': 50.09103987929191, 'cover': 47.38883632923368, 'overlap': 57.86187322611163})
## rsvp-ai/bertserini-bert-base-squad, this is finetuned based on bert-base-uncased
(0.5, {'exact_match': 40.179754020813625, 'f1': 47.828056659017584, 'recall': 49.517951036176, 'precision': 48.3495034100538, 'cover': 45.50614947965941, 'overlap': 56.20624408703879})
```

## Replication Log

+ Results replicated by [@MXueguang](https://github.com/MXueguang) on 2020-10-07 (commit [`9b670a3`](https://github.com/MXueguang/bertserini/commit/9b670a3942d24eb0188d55a140342257407f9c52)) (Tesla P40)
+ Results replicated by [@amyxie361](https://github.com/amyxie361) on 2022-03-02 (V100)