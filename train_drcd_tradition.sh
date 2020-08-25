SQUAD_DIR=/data/y247xie/00_data/DRCD

python run_squad_new.py \
  --model_type bert \
  --model_name_or_path bert-base-chinese \
  --do_train \
  --do_lower_case \
  --train_file $SQUAD_DIR/DRCD_training.json \
  --predict_file $SQUAD_DIR/DRCD_dev.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./tmp/drcd_base_tradition/

