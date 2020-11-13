import os

import numpy as np

import torch
import argparse
from tqdm import tqdm

from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from transformers import ElectraTokenizer, ElectraForSequenceClassification, ElectraConfig
from transformers import XLMRobertaTokenizer, XLMRobertaConfig, XLMRobertaForSequenceClassification
from transformers import DistilBertConfig, DistilBertForSequenceClassification
from transformers import BertConfig, BertForSequenceClassification

from tokenization_kobert import KoBertTokenizer

from dataloader import data_loader
from config import Config
import sklearn.metrics

args = argparse.ArgumentParser()
args.add_argument("--train_id", type=str, default=None)
args.add_argument("--mode", type=str, default='val')
args.add_argument("--config_path", type=str, default=None)

args = args.parse_args()
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(bert_config_class, bert_model_class, bert_model_name, num_classes=4):
    label_lst = [i for i in range(num_classes)]
    num_labels = num_classes

    config = bert_config_class.from_pretrained(bert_model_name,
                                               num_labels=num_labels,
                                               finetuning_task='nsmc',
                                               id2label={str(i): label for i, label in enumerate(label_lst)},
                                               label2id={label: i for i, label in enumerate(label_lst)})
    model = bert_model_class.from_pretrained(bert_model_name, config=config)
    model.to(device)
    
    return model


def evaluate(data_path, model_path, bert_model_info, result_path, batch_size, max_seq_len, is_segment_id):
    bert_model_name = bert_model_info['bert_model_name']
    bert_config_class = bert_model_info['bert_config_class']
    bert_tokenizer_class = bert_model_info['bert_tokenizer_class']
    bert_model_class = bert_model_info['bert_model_class']

    # get data loader
    tokenizer = bert_tokenizer_class.from_pretrained(bert_model_name)

    param = {
        "root": data_path,
        "batch_size": batch_size,
        "tokenizer": tokenizer,
        "max_seq_len": max_seq_len,
        "result_path": result_path,
        "is_segment_id": is_segment_id
    }
    if args.mode == 'val':
        dataloader = data_loader(**param, phase='validate')
    else:
        dataloader = data_loader(**param, phase='test')

    model = load_model(bert_config_class, bert_model_class, bert_model_name)
    state = torch.load(os.path.join(model_path))
    model.load_state_dict(state['model'])

    preds = None
    for batch in tqdm(dataloader, desc="evaluating"):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            if not is_segment_id:
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'labels': batch[2]
                }
            else:
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    "token_type_ids": batch[2],
                    'labels': batch[3]
                }

            outputs = model(**inputs)
            tmp_eval_loss, logits = outputs[:2]
            logits = torch.sigmoid(logits)

            if preds is None:
                preds = logits.detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
    del model

    return preds


def read_test_file(file_name):
    with open(file_name, 'r') as f:
        tag_list = []
        for line in f.readlines()[1:]:
            v = line.strip().split('\t')
            tag_list.append(int(v[1]))
    return tag_list


if __name__ == '__main__':
    config = Config()
    logits = None

    if args.mode == 'val':
        labels = read_test_file(os.path.join(config.base_dir, config.data_dir, 'validate.tsv'))

    model_infos = [
        {
            "train_id": 'kcbert',
            "bert_model_info": {'bert_model_name': 'beomi/kcbert-base',
                                'bert_config_class': AutoConfig,
                                'bert_tokenizer_class': AutoTokenizer,
                                'bert_model_class': BertForSequenceClassification},
            "model_weight_files": ['epoch_5.pth', 'epoch_6.pth', 'epoch_7.pth']
        },
        {
            "train_id": 'electra',
            "bert_model_info": {'bert_model_name': 'monologg/koelectra-base-v3-discriminator',
                                'bert_config_class': ElectraConfig,
                                'bert_tokenizer_class': ElectraTokenizer,
                                'bert_model_class': ElectraForSequenceClassification},
            "model_weight_files": ['epoch_5.pth', 'epoch_6.pth', 'epoch_7.pth']
        },
        {
            "train_id": 'kcbert_v2',
            "bert_model_info": {'bert_model_name': 'jason9693/soongsil-roberta-base',
                                'bert_config_class': AutoConfig,
                                'bert_tokenizer_class': AutoTokenizer,
                                'bert_model_class': AutoModelForSequenceClassification},
            "model_weight_files": ['epoch_5.pth', 'epoch_6.pth', 'epoch_7.pth']
        },
        {
            "train_id": 'xmlroberta',
            "bert_model_info": {'bert_model_name': 'xlm-roberta-base',
                                'bert_config_class': XLMRobertaConfig,
                                'bert_tokenizer_class': XLMRobertaTokenizer,
                                'bert_model_class': XLMRobertaForSequenceClassification},
            "model_weight_files": ['epoch_5.pth', 'epoch_6.pth', 'epoch_7.pth']
        },
        {
            "train_id": 'distilkobert',
            "bert_model_info": {'bert_model_name': 'monologg/distilkobert',
                                'bert_config_class': DistilBertConfig,
                                'bert_tokenizer_class': KoBertTokenizer,
                                'bert_model_class': DistilBertForSequenceClassification},
            "model_weight_files": ['epoch_5.pth', 'epoch_6.pth', 'epoch_7.pth']
        },
        {
            "train_id": 'kobert',
            "bert_model_info": {'bert_model_name': 'monologg/kobert',
                                'bert_config_class': BertConfig,
                                'bert_tokenizer_class': KoBertTokenizer,
                                'bert_model_class': BertForSequenceClassification},
            "model_weight_files": ['epoch_5.pth', 'epoch_6.pth', 'epoch_7.pth']
        }
    ]

    # model_infos = [
    #     {
    #         "train_id": 'baseline',
    #         "bert_model_info": {'bert_model_name': 'beomi/kcbert-base',
    #                             'bert_config_class': AutoConfig,
    #                             'bert_tokenizer_class': AutoTokenizer,
    #                             'bert_model_class': BertForSequenceClassification},
    #         "model_weight_files": ['epoch_4.pth']
    #     },
    #     {
    #         "train_id": 'electra_test',
    #         "bert_model_info": {'bert_model_name': 'monologg/koelectra-base-v3-discriminator',
    #                             'bert_config_class': ElectraConfig,
    #                             'bert_tokenizer_class': ElectraTokenizer,
    #                             'bert_model_class': ElectraForSequenceClassification},
    #         "model_weight_files": ['epoch_5.pth']
    #     },
    #     {
    #         "train_id": 'kcbert_v2',
    #         "bert_model_info": {'bert_model_name': 'jason9693/soongsil-roberta-base',
    #                             'bert_config_class': AutoConfig,
    #                             'bert_tokenizer_class': AutoTokenizer,
    #                             'bert_model_class': AutoModelForSequenceClassification},
    #         "model_weight_files": ['epoch_5.pth', 'epoch_6.pth']
    #     }
    # ]

    for info in model_infos:
        bert_model_name = info["bert_model_info"]["bert_model_name"]
        train_id = info["train_id"]
        bert_model_info = info["bert_model_info"]
        model_weight_files = info["model_weight_files"]
        is_segment_id = True \
            if bert_model_name not in ['xlm-roberta-base', 'monologg/distilkobert', 'jason9693/soongsil-roberta-base'] \
            else False

        for model_weight_file in model_weight_files:
            data_dir = os.path.join(config.base_dir, config.data_dir)
            result_dir = os.path.join(config.base_dir, config.result_dir, train_id)
            model_path = os.path.join(result_dir, model_weight_file)
            batch_size = config.batch_size
            max_seq_len = config.max_seq_len

            inputs = {
                'data_path': data_dir,
                'model_path': model_path,
                'bert_model_info': bert_model_info,
                'result_path': result_dir,
                'batch_size': batch_size,
                'max_seq_len': max_seq_len,
                'is_segment_id': is_segment_id
            }
            if logits is not None:
                logits += evaluate(**inputs)
            else:
                logits = evaluate(**inputs)

            print('Add ', train_id, ' ', model_weight_file)
            preds = np.argmax(logits, axis=1)

            with open(os.path.join(config.base_dir, 'prediction.tsv'), "w", encoding="utf-8") as f:
                for pred in preds:
                    f.write("{}\n".format(pred))

            if args.mode == 'val':
                f1_tag = sklearn.metrics.f1_score(labels, preds, average='weighted')
                f1_tag2 = sklearn.metrics.f1_score(labels, preds, average='macro')
                f1_tag3 = sklearn.metrics.f1_score(labels, preds, average=None)

                print('f1_tag : ' + str(f1_tag))
                print('f1_tag2 : ' + str(f1_tag2))
                print(f1_tag3)
                print('')