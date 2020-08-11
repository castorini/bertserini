DIR=./results/drcd_base
FILE=test_inference_drcd.json
DATA_PATH=/data/yqxie/00_data/DRCD/DRCD_dev.json

python aggregate.py \
    --eval_data ${DATA_PATH} \
    --search_file ${DIR}/${FILE} \
    --output_path ${DIR} \
    --dataset drcd

