SQUAD_DIR=~/00_data/DRCD

python ./search.py \
    --para_num 10 \
    --index_path index/lucene-index.wiki_zh_paragraph_with_title_0.6.0.pos+docvectors \
    --model_type bert \
    --model_name_or_path ./tmp/debug_bert_base_drcd \
    --do_eval \
    --do_lower_case \
    --max_seq_length 384 \
    --doc_stride 128 \
    --predict_file $SQUAD_DIR/DRCD_dev.json \
    --eval_batch_size=32 \
    --output_fn test_inference_drcd.json \
    --output_dir bert_base_DRCD
