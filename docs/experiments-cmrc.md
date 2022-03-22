# Bertserini: Baseline on CMRC QA (in Chinese)

1. Clone the repo with ```git clone https://github.com/castorini/bertserini.git```
2. ```pip install -r requirements.txt --find-links https://download.pytorch.org/whl/torch_stable.html```

## Download PreBuilt Wikipedia Index

We have indexed the 2018 Wikipedia Chinese dump. You can download the prepared index here:
```
cd indexes
wget https://rgw.cs.uwaterloo.ca/JIMMYLIN-bucket0/pyserini-indexes/lucene-index.zhwiki-20181201-paragraphs.tar.gz
tar -xvf lucene-index.zhwiki-20181201-paragraphs.tar.gz
rm lucene-index.zhwiki-20181201-paragraphs.tar.gz
cd ..
```
It contains the indexed 20181201 Chinese Wikipedia dump with Anserini.

## Download the pre-trained models

We have uploaded the finetuned checkpoints to the huggingface models: \
[bertserini-bert-base-cmrc](https://huggingface.co/rsvp-ai/bertserini-bert-base-cmrc)


To run our finetuned model, just set ```--model_name_or_path rsvp-ai/<MODEL_NAME>```.  
For example: ```--model_name_or_path rsvp-ai/bertserini-bert-base-cmrc```.

# Inferencing and evaluating

## Prepare the datasets:

```
cd data
wget https://worksheets.codalab.org/rest/bundles/0xb70e5e281fcd437d9aa8f1c4da107ae4/contents/blob/
mv index.html cmrc2018_dev.json
wget https://worksheets.codalab.org/rest/bundles/0x72252619f67b4346a85e122049c3eabd/contents/blob/
mv index.html cmrc2018_dev_squad.json
cd ..
```

## Inferencing CMRC under the open-domain setting
For `rsvp-ai/bertserini-bert-base-cmrc`
```
python -m bertserini.experiments.inference --dataset_path data/cmrc2018_dev_squad.json \
                                           --index_path indexes/lucene-index.zhwiki-20181201-paragraphs \
                                           --model_name_or_path rsvp-ai/bertserini-bert-base-cmrc \
                                           --output prediction/cmrc2018_pred.json \
                                           --support_no_answer False \
                                           --topk 10 \
                                           --language zh \
                                           --strip_accents

```

## Evaluation

```
mkdir temp
python -m bertserini.experiments.evaluate --eval_data data/cmrc2018_dev.json \
                                          --search_file prediction/cmrc2018_pred.json \
                                          --output_path temp \
                                          --dataset cmrc
                                          
```

Expected results:

```
## rsvp-ai/bertserini-bert-base-cmrc, this is bert-base-chinese finetuned on the chinese reading comprehension dataset(CMRC)
(0.5, {'f1_score': 68.42519791441855, 'exact_match': 51.94159676918298, 'total_count': 3219, 'skip_count': 0})
```

## Replication Log

+ Results replicated by [@MXueguang](https://github.com/MXueguang) on 2020-10-07 (commit [`9b670a3`](https://github.com/MXueguang/bertserini/commit/9b670a3942d24eb0188d55a140342257407f9c52)) (Tesla P40)
+ Results replicated by [@amyxie361](https://github.com/amyxie361) on 2022-03-10 (V100)