import os
import math
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import argparse
import random
import logging

from transformers import AutoTokenizer, AutoModel, AutoConfig
from bayes_opt import BayesianOptimization

from model import Trainer
from dataloader import data_loader
from config import Config

'''
data
 \_ train.tsv
 \_ test.tsv
 \_ validate.tsv(생성 필요)
 \_ pred.tsv(train시 validate 예측값, test시 test 예측값 저장)

model
 \_ 1.pth, ...

 evaluation.py를 통해 예측값의 f1 score 확인 가능

# 데이터, 모델 경로 수정 바랍니다.
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
    print(model.device)
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