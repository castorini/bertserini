SQUAD_DIR=/data/yqxie/00_data/cmrc2018/data

python run_squad_new.py \
  --model_type bert \
  --model_name_or_path ./tmp/bert_wwm_cmrc/checkpoint-3500 \
  --do_eval \
  --train_file $SQUAD_DIR/cmrc_train_to_squad.json \
  --predict_file $SQUAD_DIR/cmrc_dev_to_squad.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./tmp/debug_cmrc/

