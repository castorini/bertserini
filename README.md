# BERTserini

## Install dependencies

```
conda create -n bertserini
conda activate bertserini
conda install tqdm
pip install pyserini
pip install transformers 
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html # or install torch according to your cuda version
pip install tensorboardX
```

## Get index and lib ready

download the prepared index and lib:
```
wget ftp://72.143.107.253/BERTserini/index.zip
wget ftp://72.143.107.253/BERTserini/lib.zip
````
Inside contains two zips: index.zip and lib.zip
index.zip contains the indexed latest wikipedia dump with Answerini.
lib.zip contains the compiled .jar files needed for searching paragraphs using Anserini.
After unzip these files, put them under the root path of this repo and then you are ready to go.
Take the folloing folder structure as example:
```
bertserini
+--- index
|    +--- lucene-index.enwiki-20180701-paragraphs
|    |    +--- ...
+--- lib
|    +--- *.jar ...
+--- otherfiles under this repo
```

## Download datasets:

```
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json
```

## Run training
Please set the correct parameters in the following script and then run.
```
bash train.sh
```
This script will run the training for BERT on SQuAD dataset.
It will generate checkpoints under ./tmp/debug_squad/

## Run inferencing SQuAD under open-domain setting
Set the ckpt information in the below script, according to the path after training.
```
bash inference.sh
```
It will generate inference results on SQuAD 1.1, under the path of ./results/

## evaluation
Set the result path according to the inference result path
```
bash eval.sh
```
This will first automatically select the parameter to aggregate paragraph score (from Pyserini) and phrase score (from BERT), and finally select the best parameter and print the evaluation matrixs.
```
# expected result:
(0.4, {'exact_match': 40.89877010406812, 'f1': 48.827808932780215, 'recall': 50.644587225343955, 
'precision': 49.308238592369754, 'cover': 46.87795648060549, 'overlap': 57.28476821192053})
```
