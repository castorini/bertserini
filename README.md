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

download the index and lib from our dropbox:

## run training

bash train.sh

## run inference under open-domain setting

bash test_search.sh
