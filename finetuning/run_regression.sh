#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: ./run_regression.sh path/to/labels.csv path/to/embedding.pkl"
  exit 1
fi

# 🔹 입력 인자
embedding_dataset=$1     # CSV 파일 경로 (labels 포함)
embedding_pkl=$2         # 임베딩이 저장된 .pkl 파일 경로
device="cuda:0"          # 사용할 GPU 디바이스

# 🔹 하이퍼파라미터 설정
batch_size=256
epochs=100
k_folds=5
input_dim=256
lr=0.005
weight_decay=1e-3

# 🔹 로그 저장 디렉토리 생성
log_dir="regression_logs"
mkdir -p $log_dir

# 🔹 실행
python regression.py \
  --embedding_dataset "$embedding_dataset" \
  --embedding_pkl "$embedding_pkl" \
  --device "$device" \
  --batch_size "$batch_size" \
  --epochs "$epochs" \
  --k_folds "$k_folds" \
  --input_dim "$input_dim" \
  --lr "$lr" \
  --weight_decay "$weight_decay" \
  > "$log_dir/kfold_regression.log" 2>&1
