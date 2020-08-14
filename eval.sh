DIR=./results/bert_base_squad
FILE=test_inference.json
DATA_PATH=/data/yqxie/00_data/squad_v1.1/dev-v1.1.json

python aggregate.py \
    --eval_data ${DATA_PATH} \
    --search_file ${DIR}/${FILE} \
    --output_path ${DIR} \
    --dataset squad

