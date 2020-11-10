import os
import easydict


class Config:
    BASE_DIR = '/home/ubuntu/aikorea/sbs'
    # 학습 수행 정보 (eval 정보, model checkpoint, 각종 cache 파일 등) 을 저장할 폴더 *해당 경로가 없으면 폴더 자동 생성
    TRAIN_DIR = os.path.join(BASE_DIR, 'train/task156_kcbert_base')

    # Train 모델 Checkpoint 저장할 경로
    MODEL_SAVE_DIR = os.path.join(TRAIN_DIR, 'model')

    # huggingface BERT 모델명
    BERT_MODEL_NAME ='beomi/kcbert-base'

    # 데이터셋 저장 경로
    DATASET_PATH = os.path.join(BASE_DIR, 'data')

    # test 결과 저장 경로
    PREDICT_DIR = os.path.join(BASE_DIR, 'result')
    # train/test
    mode = 'train'

    #학습정보
    args = easydict.EasyDict(
        {
            'num_classes': 4,
            'lr': 0.0001,
            'cuda': True,
            'num_epochs': 4,
            'print_iter': 300,
            'model_name': '3.pth',
            'seed': 42,
            'prediction_file': os.path.join(PREDICT_DIR, 'prediction.tsv'),
            'batch': 45,
            'mode': mode,
            'gradient_accumulation_steps': 1,
            'logging_steps': 983,
            'save_steps': 983
        }
    )