SQUAD_DIR=/data/y247xie/00_data/squad/v1.1

python run_squad_new.py \
  --model_type electra \
  --model_name_or_path google/electra-base-discriminator \
  --do_train \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 8 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./tmp/electra-base-squad/

