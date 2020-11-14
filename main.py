import os

import datetime
import numpy as np

import torch
import random
import logging
import argparse

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import ElectraTokenizer, ElectraForSequenceClassification, ElectraConfig
from transformers import XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification
from transformers import DistilBertConfig, DistilBertForSequenceClassification
from transformers import BertConfig, BertForSequenceClassification

from tokenization_kobert import KoBertTokenizer

from bayes_opt import BayesianOptimization

from model import Trainer
from dataloader import data_loader
from config import Config

args = argparse.ArgumentParser()
args.add_argument("--num_epochs", type=int, default=0)
args.add_argument("--alpha", type=float, default=None)
args.add_argument("--gamma", type=float, default=None)
args.add_argument("--train_id", type=str, default=None)
args.add_argument("--mode", type=str, default=None)
args.add_argument("--use_bayes_opt", type=bool, default=False)
args.add_argument("--use_swa", type=bool, default=True)
args.add_argument("--config_path", type=str, default="config.json")
args.add_argument("--base_dir", type=str, default=None)


args = args.parse_args()

model_infos = {
        "beomi/kcbert-base": {
            'bert_config_class': AutoConfig,
            'bert_tokenizer_class': AutoTokenizer,
            'bert_model_class': BertForSequenceClassification
        },
        "beomi/kcbert-large": {
            'bert_config_class': AutoConfig,
            'bert_tokenizer_class': AutoTokenizer,
            'bert_model_class': BertForSequenceClassification
        },
        "monologg/koelectra-base-v3-discriminator": {
            'bert_config_class': ElectraConfig,
            'bert_tokenizer_class': ElectraTokenizer,
            'bert_model_class': ElectraForSequenceClassification
        },
        "jason9693/soongsil-roberta-base": {
            'bert_config_class': AutoConfig,
            'bert_tokenizer_class': AutoTokenizer,
            'bert_model_class': AutoModelForSequenceClassification
        },
        "xlm-roberta-base": {
            'bert_config_class': XLMRobertaConfig,
            'bert_tokenizer_class': XLMRobertaTokenizer,
            'bert_model_class': XLMRobertaForSequenceClassification
        },
        "monologg/distilkobert": {
            'bert_config_class': DistilBertConfig,
            'bert_tokenizer_class': KoBertTokenizer,
            'bert_model_class': DistilBertForSequenceClassification
        },
        "monologg/kobert": {
            'bert_config_class': BertConfig,
            'bert_tokenizer_class': KoBertTokenizer,
            'bert_model_class': BertForSequenceClassification
        }
    }

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
    config = Config(args.config_path)
    if args.mode:
        config.mode = args.mode
    if args.train_id:
        config.train_id = args.train_id
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.base_dir:
        config.base_dir = args.base_dir

    config.use_bayes_opt = args.use_bayes_opt
    config.use_swa = args.use_swa

    train_path = os.path.join(config.base_dir, config.train_dir, config.train_id)
    result_path = os.path.join(config.base_dir, config.result_dir, config.train_id)
    data_path = os.path.join(config.base_dir, config.data_dir)

    if not os.path.isdir(train_path):
        os.mkdir(train_path)

    if not os.path.isdir(result_path):
        os.mkdir(result_path)

    init_logger(os.path.join(result_path, 'log.txt'))
    set_seed(config)

    model_classes = model_infos[config.bert_model_name]
    # get data loader
    tokenizer = model_classes['bert_tokenizer_class'].from_pretrained(config.bert_model_name)

    is_segment_id = True \
        if config.bert_model_name not in ['xlm-roberta-base', 'monologg/distilkobert', 'jason9693/soongsil-roberta-base'] \
        else False
    param = {
        "root": data_path,
        "batch_size": config.batch_size,
        "tokenizer": tokenizer,
        "max_seq_len": config.max_seq_len,
        "result_path": result_path,
        "is_segment_id": is_segment_id
    }
    train_dataloader = data_loader(**param, phase='train')
    validate_dataloader = data_loader(**param, phase='validate')
    test_dataloader = data_loader(**param, phase='test')


    # create model config 확인
    model = Trainer(config, model_classes['bert_config_class'], model_classes['bert_model_class'], \
                    train_dataloader, validate_dataloader, test_dataloader)

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
    bayes_optimizer.maximize(init_points=3, n_iter=10, acq='ei', xi=0.01)
else:
    train_result = main(alpha=args.alpha, gamma=args.gamma)
    logging.info('Train Result %f' % (train_result))
