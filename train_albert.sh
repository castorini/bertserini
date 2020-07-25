SQUAD_DIR=/data/y247xie/00_data/squad/v1.1

python run_squad_new.py \
  --model_type albert \
  --model_name_or_path albert-base-v2 \
  --do_train \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./tmp/albert-base-v2-squad/

