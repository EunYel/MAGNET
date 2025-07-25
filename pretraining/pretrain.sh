#!/bin/bash

INPUT_FILE=$1
 
# 입력 파일이 제공되지 않은 경우 에러 메시지 출력
if [ -z "$INPUT_FILE" ]; then
  echo "❌ Error: Please provide the input file path."
  echo "Usage: ./run_pretrain.sh path/to/preprocessed.pkl"
  exit 1
fi

python pretrain.py $INPUT_FILE \
  --node_dim 768 \
  --edge_dim 1 \
  --num_blocks 4 \
  --num_heads 8 \
  --last_average \
  --model_dim 256 \
  --temperature 0.2 \
  --lr 1e-4 \
  --batch_size 32 \
  --train_epoch 30
