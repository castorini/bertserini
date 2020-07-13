SQUAD_DIR=~/00_data/squad_v1.1/

python ./search.py \
    --para_num 100 \
    --index_path index/lucene-index.enwiki-20180701-paragraphs \
    --model_type bert \
    --model_name_or_path twmkn9/bert-base-uncased-squad2 \
    --do_eval \
    --do_lower_case \
    --max_seq_length 384 \
    --doc_stride 128 \
    --predict_file $SQUAD_DIR/dev-v1.1.json \
    --eval_batch_size=32 \
    --output_fn test_inference.json \
    --output_dir test_inference
