SQUAD_DIR=/data/yqxie/00_data/squad_v1.1

python run_squad_new.py \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --do_train \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_DIR/dev-v1.1.json \
  --per_gpu_train_batch_size 12 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir ./tmp/roberta-base-squad/

