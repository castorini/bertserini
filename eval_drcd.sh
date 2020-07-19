DIR=./results/bert_wwm_drcd
FILE=test_inference.json
DATA_PATH=/data/yqxie/00_data/DRCD/DRCD_dev.json

python aggregate.py \
    --eval_data ${DATA_PATH} \
    --search_file ${DIR}/${FILE} \
    --output_path ${DIR} \
    --dataset drcd

