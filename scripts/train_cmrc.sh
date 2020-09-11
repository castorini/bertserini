SQUAD_DIR=/data/y247xie/00_data/cmrc2018/data

python run_squad.py \
  --model_type bert \
  --model_name_or_path bert-base-chinese \
  --do_train \
  --do_lower_case \
  --train_file $SQUAD_DIR/CMRC_train_to_squad.json \
  --predict_file $SQUAD_DIR/CMRC_dev_to_squad.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --version_2_with_negative \
  --output_dir ./tmp/cmrc_base

