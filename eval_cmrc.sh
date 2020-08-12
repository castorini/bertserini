DIR=./results/cmrc_electra
FILE=test_inference_cmrc_electra.json
DATA_PATH=/data/yqxie/00_data/cmrc2018/data/cmrc2018_dev.json

python aggregate.py \
    --eval_data ${DATA_PATH} \
    --search_file ${DIR}/${FILE} \
    --output_path ${DIR} \
    --dataset cmrc

