# Training for BERT on the SQuAD dataset

The following script will train the SQuAD style training dataset on BERT 
and then evaluate the checkpoints on corresponding development set.

### 1. Download SQuAD train set

```
cd data
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

### 2. Run training script

```
python -m bertserini.train.run_squad  --model_type bert \
                                      --model_name_or_path bert-base-uncased \
                                      --do_train \
                                      --do_lower_case \
                                      --train_file data/train-v1.1.json \
                                      --predict_file data/dev-v1.1.json \
                                      --per_gpu_train_batch_size 12 \
                                      --learning_rate 3e-5 \
                                      --num_train_epochs 2.0 \
                                      --max_seq_length 384 \
                                      --doc_stride 128 \
                                      --output_dir models/bert_base_squad/
```

## For CMRC2018 dataset 

### 1. Download CMRC train set in squad format 

```
cd data
wget https://worksheets.codalab.org/rest/bundles/0x15022f0c4d3944a599ab27256686b9ac/contents/blob/
mv index.html cmrc2018_train_squad.json
wget https://worksheets.codalab.org/rest/bundles/0x72252619f67b4346a85e122049c3eabd/contents/blob/
mv index.html cmrc2018_dev_squad.json
```

### 2. Run training script

```
python -m bertserini.train.run_squad  --model_type bert \
                                      --model_name_or_path bert-base-chinese \
                                      --do_train \
                                      --do_lower_case \
                                      --train_file data/cmrc2018_train_squad.json \
                                      --predict_file data/cmrc2018_dev_squad.json \
                                      --per_gpu_train_batch_size 12 \
                                      --learning_rate 3e-5 \
                                      --num_train_epochs 2.0 \
                                      --max_seq_length 384 \
                                      --doc_stride 128 \
                                      --version_2_with_negative \
                                      --output_dir models/bert_base_cmrc/
```

