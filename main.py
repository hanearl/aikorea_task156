import os

import datetime
import numpy as np

import torch
import random
import logging
import argparse

from transformers import AutoTokenizer, AutoModel, AutoConfig
from bayes_opt import BayesianOptimization

from model import Trainer
from dataloader import data_loader
from config import Config

'''
테스트 리스트
1. baseline 설정
2. focal loss
3. 전처리 추가하기
4. SWA 적용해보기
4. kcbert-large, electra-small, electra-base 등 모델 테스트 (large는 별로!)
5. bert 레이어 별 cls 결과 및 앙상블 결과

고민거리
1. confusion matrix 어느 tag가 정확도가 안나오나
2. 틀린 case 들 모아서 분석하기
3. 너무 안좋은 데이터는 버리는게 좋은가?
4. 어떤 데이터가 학습을 방해하는
'''
args = argparse.ArgumentParser()
args.add_argument("--num_epochs", type=int, default=0)
args.add_argument("--train_id", type=str, default=None)
args.add_argument("--mode", type=str, default=None)
args.add_argument("--use_bayes_opt", type=bool, default=False)
args.add_argument("--use_preprocess", type=bool, default=False)
args.add_argument("--use_swa", type=bool, default=False)

args = args.parse_args()


def init_logger(filename):
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        filename=filename)


def set_seed(args):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def main(alpha=None, gamma=None):
    config = Config("config.json")
    if args.mode:
        config.mode = args.mode
    if args.train_id:
        config.train_id = args.train_id
    if args.num_epochs:
        config.num_epochs = args.num_epochs

    config.use_bayes_opt = args.use_bayes_opt
    config.use_preprocess = args.use_preprocess
    config.use_swa = args.use_swa

    train_path = os.path.join(config.base_dir, config.train_dir, config.train_id)
    result_path = os.path.join(config.base_dir, config.result_dir, config.train_id)
    data_path = os.path.join(config.base_dir, config.data_dir)

    if not os.path.isdir(train_path):
        os.mkdir(train_path)

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    init_logger(os.path.join(config.result_dir, config.train_id, 'log.txt'))
    set_seed(config)

    # get data loader
    tokenizer = AutoTokenizer.from_pretrained(config.bert_model_name)

    param = {"root": data_path, "batch_size": config.batch_size, "tokenizer": tokenizer, "config": config}
    train_dataloader = data_loader(**param, phase='train')
    validate_dataloader = data_loader(**param, phase='validate')
    test_dataloader = data_loader(**param, phase='test')

    # create model config 확인
    model = Trainer(config, train_dataloader, validate_dataloader, test_dataloader)

    if config.mode == 'train':
        result = model.train(alpha=alpha, gamma=gamma)
    elif config.mode == 'test':
        model.load_model(config.model_weight_file)
        result = model.evaluate('test')

    del model
    return result


if args.use_bayes_opt:
    bayes_optimizer = BayesianOptimization(main,
                                   {
                                       'alpha': (0.1, 0.95),
                                       'gamma': (0.1, 5)
                                   },
                                   random_state=42)
    bayes_optimizer.maximize(init_points=5, n_iter=20, acq='ei', xi=0.01)
else:
    train_result = main()
    logging.info('Train Result %f' % (train_result))