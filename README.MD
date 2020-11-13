**설정 사용법**

config.json

  - "num_classes": 레이블 개수
  - "num_epochs": 학습할 epoch 수
  - "model_weight_file": "epoch_4.pth",
  - "bert_model_name": huggingface에서 로드할 모델 이름
  
|   모델명   |      설정      |
|----------|:-------------:|
| kcBERT |  beomi/kcbert-base |
| kcBERT-large | beomi/kcbert-large|
| kcBERT_v2 | jason9693/soongsil-roberta-base|
| Electra | monologg/koelectra-base-v3-discriminator |
| xmlroberta | xlm-roberta-base |
| distilkobert | monologg/distilkobert |
| koBERK | monologg/kobert |


  - "base_dir": 프로젝트 base 경로
  - "train_dir": os.path.join(base_dir, train_dir) 경로로 생긴다
  - "data_dir": train.tsv, validate.tsv, test.tsv가 있는 경로, os.path.join(base_dir, data_dir) 경로로 생긴다
  - "result_dir": 결과물이 모이는 경로, 아래 train_id와 같이 쓰임 , os.path.join(base_dir, result_dir) 경로로 생긴다
  - "train_id": 모델파일, 로그 등이 저장되는 경로, os.path.join(base_dir, result_dir, train_id) 경로로 생긴다 
  - "prediction_file": eval시 prediction.tsv 생기는 경로 (풀 경로를 적어줄 것)
  - "batch_size": 배치사이즈
  - "mode": train : 학습, test : eval (결과물이 나온다)



**실행법**

python main.py --config_path=config.json --use_swa=True 

