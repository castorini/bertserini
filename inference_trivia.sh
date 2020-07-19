SQUAD_DIR=/data/yqxie/00_data/triviaQA

python ./search.py \
    --para_num 10 \
    --index_path index/lucene-index.enwiki-20180701-paragraphs \
    --model_type bert \
    --model_name_or_path ./tmp/debug_bert_base_trivia/ \
    --do_eval \
    --do_lower_case \
    --max_seq_length 384 \
    --doc_stride 128 \
    --predict_file $SQUAD_DIR/squad-wikipedia-dev-4096.json \
    --eval_batch_size=32 \
    --output_fn test_inference_trivia.json \
    --output_dir bert_base_trivia
