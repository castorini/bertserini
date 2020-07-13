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
download the SQuAD datasets

## Run training
Please set the correct parameters in the following script and then run.
```
bash train.sh
```

## Run inferencing SQuAD under open-domain setting

bash test_search.sh

## evaluation
