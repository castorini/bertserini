# bertserini

## install dependencies

```
conda create -n bertserini
conda activate bertserini
conda install tqdm
pip install transformers 
pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html # or install torch according to your cuda version
pip install tensorboardX
```

## get index and lib ready

download the index and lib:
```
wget ftp://72.143.107.253/BERTserini/bertserini.zip
```
Inside contains two folder: index and lib, after unzip, put them under the root path of this folder and then you are ready to go.

## run training

bash train.sh

## run inference under open-domain setting

bash test_search.sh
