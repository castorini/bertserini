DIR=./results
FILE=test_inference_xlnet.json
DATA_PATH=/data/y247xie/00_data/squad/v1.1/dev-v1.1.json

python aggregate.py \
    --eval_data ${DATA_PATH} \
    --search_file ${DIR}/${FILE} \
    --output_path ${DIR} \
    --dataset squad

