#!/bin/bash

# 인자 확인
if [ "$#" -ne 2 ]; then
  echo "Usage: ./run_classification.sh path/to/labels.csv path/to/embedding.pkl"
  exit 1
fi

embedding_dataset=$1
embedding_pkl=$2
device="cuda:0"

# 하이퍼파라미터
batch_size=256
epochs=100
k_folds=5
input_dim=1024

# 로그 디렉토리
log_dir="classification_logs"
mkdir -p $log_dir

# 실행
python classification.py \
  --embedding_dataset "$embedding_dataset" \
  --embedding_pkl "$embedding_pkl" \
  --device "$device" \
  --batch_size "$batch_size" \
  --epochs "$epochs" \
  --k_folds "$k_folds" \
  --input_dim "$input_dim" \
  > "$log_dir/classification_kfold.log" 2>&1
