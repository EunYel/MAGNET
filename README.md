# MAGNET: Multi-view Aggregation of Graphs for Neural Embedding of Topologies

## MAGNET Workframe(Pretraining → Embedding → Downstream)

MAGNET 프레임워크를 사용하여 분자 데이터를 사전학습하고, 임베딩을 추출한 뒤, classification 또는 regression downstream task에 사용할 수 있습니다.

---

## 1. Preprocessing

### Command

```bash
cd ./MAGNET/preprocessing
python preprocess.py \[path/to/your_dataset.csv\]
```

> ⚠️ **주의:**
> 데이터셋 내 SMILES 컬럼명이 `smiles`가 아닐 경우, 코드 내에서 `"smiles"`로 변경해주어야 합니다.

### Output

* `preprocessed_MetaData/` 폴더에 `.pkl` 파일 생성
* 포함된 항목:

  * `"filtered_data"`: SMILES와 라벨 포함된 필터링된 DataFrame
  * `"all_graphs"`: Graph Transformer 입력용 그래프 리스트

---

## 2. Pretraining

### Command

```bash
cd ./MAGNET/pretraining
chmod +x run_pretrain.sh
./run_pretrain.sh path/to/preprocessed_dataset.pkl
```
> 💡 참고: run_pretrain.sh 파일에서 lr, batch_size, temperature 등의 하이퍼파라미터 수정 가능

### Output

* `outputs/` 폴더에 다음 파일 생성:

  * `pretrain_test_model.pt`: 사전학습된 Graph Transformer 가중치
  * `loss_curve.png`: 학습 손실 시각화


## 3. Finetuning (Downstream)

사전학습된 모델을 기반으로 downstream 데이터셋의 embedding을 생성한 후, classification 또는 regression task를 수행합니다.



### 3-1. Embedding 추출

```bash
cd ./MAGNET/finetuning

python extract_embedding.py \
  --input_file path/to/downstream_dataset.csv \
  --pt_file outputs/pretrain_model.pt
```

### Output

* `finetuning_embedding/embedding_{dataset_name}.pkl`

## 

### 3-2. Downstream Tasks

**⚠️주의 : downstream dataset에서 사용할 target 컬럼은 반드시 `labels`로 변경되어 있어야 합니다.**


#### (1) Classification

```bash
chmod +x run_classification.sh
./run_classification.sh path/to/labels.csv path/to/embedding.pkl
```


* Output: `classification_logs/kfold_classification.log`

> 💡 참고: run_classification.sh 파일에서 batch_size, epochs, input_dim 등의 파라미터 조정 가능

#### (2) Regression

```bash
chmod +x run_regression.sh
./run_regression.sh path/to/labels.csv path/to/embedding.pkl
```

* Output: `regression_logs/kfold_regression.log`

> 💡 참고: run_regression.sh 파일에서 batch_size, epochs, lr, input_dim 등 파라미터 수정 가능
