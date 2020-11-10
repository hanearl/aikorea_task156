import os

import datetime
import numpy as np

import torch
import random
import logging

from transformers import AutoTokenizer, AutoModel, AutoConfig

from model import Trainer
from dataloader import data_loader
from config import Config

'''
1. confusion matrix 어느 tag가 정확도가 안나오나
2. 틀린 case 들 모아서 분석하기
3. 같은조건 bert 모델별 분석해보기
4. 같은 기법 전처리 기법 달리해보기
5. bert 레이어 별 cls 결과 및 앙상블 결과

1. 너무 안좋은 데이터는 버리는게 좋은가?
2. 어떤 데이터가 학습을 방해하는
'''

def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def main(alpha, gamma):
    if not os.path.isdir(Config.TRAIN_DIR):
        os.mkdir(Config.TRAIN_DIR)

    if not os.path.isdir(Config.MODEL_SAVE_DIR):
        os.mkdir(Config.MODEL_SAVE_DIR)

    config = Config.args

    init_logger()
    set_seed(config)

    num_classes = config.num_classes
    base_lr = config.lr
    cuda = config.cuda
    num_epochs = config.num_epochs
    print_iter = config.print_iter
    model_name = config.model_name
    prediction_file = config.prediction_file
    batch = config.batch
    mode = config.mode
    test_file = os.path.join(Config.TRAIN_DIR, 'validate.tsv')

    # get data loader
    tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)

    train_dataloader = data_loader(root=Config.DATASET_PATH, phase='train', batch_size=batch, tokenizer=tokenizer)
    validate_dataloader = data_loader(root=Config.DATASET_PATH, phase='validate', batch_size=batch, tokenizer=tokenizer)
    test_dataloader = data_loader(root=Config.DATASET_PATH, phase='test', batch_size=batch, tokenizer=tokenizer)
    time_ = datetime.datetime.now()
    num_batches = len(train_dataloader)

    # create model config 확인
    model = Trainer(config, train_dataloader, validate_dataloader, test_dataloader)

    if mode == 'test':
        model.load_model(model_name)

    if mode == 'train':
        result = model.train(alpha, gamma)
    elif mode == 'test':
        result = model.evaluate('test')

    del model
    return result


main(alpha=0.44, gamma=2.5)
# bayes_optimizer = BayesianOptimization(main,
#                                {
#                                    'alpha': (0.1, 0.95),
#                                    'gamma': (0.1, 5)
#                                },
#                                random_state=42)
# bayes_optimizer.maximize(init_points=5, n_iter=30, acq='ei', xi=0.01)