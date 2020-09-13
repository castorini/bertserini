SQUAD_DIR=/data/y247xie/00_data/cmrc2018/data

python ./interactive.py \
    --para_num 10 \
    --index_path index/lucene-index.wiki_zh_paragraph_with_title_0.6.0.pos+docvectors \
    --model_type bert \
    --model_name_or_path rsvp-ai/bertserini-bert-base-cmrc \
    --do_eval \
    --do_lower_case \
    --max_seq_length 384 \
    --doc_stride 128 \
    --predict_file $SQUAD_DIR/cmrc_dev_to_squad.json \
    --eval_batch_size=10 \
    --output_dir=./demo_cmrc \
    --chinese
