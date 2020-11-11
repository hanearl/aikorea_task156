import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import AutoConfig, BertForSequenceClassification
import torch.nn.functional as F

from sklearn.metrics import f1_score, precision_recall_fscore_support

from loss import FocalLoss
from config import Config
import math

logger = logging.getLogger(__name__)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_score(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_score(preds, labels):
    return {
        "acc": simple_accuracy(preds, labels),
    }


class Trainer(object):
    def __init__(self, args, train_dataloader=None, validate_dataloader=None, test_dataloader=None):
        self.args = args
        self.train_dataloader = train_dataloader
        self.validate_dataloader = validate_dataloader
        self.test_dataloader = test_dataloader

        self.label_lst = [i for i in range(self.args.num_classes)]
        self.num_labels = self.args.num_classes

        self.config_class = AutoConfig
        self.model_class = BertForSequenceClassification

        self.config = self.config_class.from_pretrained(self.args.bert_model_name,
                                                        num_labels=self.num_labels,
                                                        finetuning_task='nsmc',
                                                        id2label={str(i): label for i, label in
                                                                  enumerate(self.label_lst)},
                                                        label2id={label: i for i, label in enumerate(self.label_lst)})
        self.model = self.model_class.from_pretrained(self.args.bert_model_name, config=self.config)
        self.optimizer = None
        self.scheduler = None

        # GPU or CPU
        self.device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"
        self.model.to(self.device)

    def train(self, alpha, gamma):
        train_dataloader = self.train_dataloader

        t_total = len(train_dataloader) * self.args.num_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        self.optimizer = optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.lr, eps=1e-8)
        self.scheduler = scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=0,
                                                                     num_training_steps=t_total)
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataloader) * self.args.batch_size)
        logger.info("  Num Epochs = %d", self.args.num_epochs)
        logger.info("  Total train batch size = %d", self.args.batch_size)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()
        self.optimizer.zero_grad()

        train_iterator = trange(int(self.args.num_epochs), desc="Epoch")

        fin_result = None
        for epoch in train_iterator:
            print(len(train_dataloader))
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()

                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3],
                          'token_type_ids': batch[2]}

                # outputs = self.model(**inputs)
                # loss = outputs[0]

                # # Custom Loss
                loss, logits = self.model(**inputs)
                logits = torch.sigmoid(logits)

                labels = torch.zeros((len(batch[3]), self.num_labels)).to(self.device)
                labels[range(len(batch[3])), batch[3]] = 1

                loss = self.criterion(logits, labels)

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()  # Update learning rate schedule

                self.model.zero_grad()
                self.optimizer.zero_grad()

                tr_loss += loss.item()
                global_step += 1
                logger.info('train loss %f', loss.item())

            logger.info('total train loss %f', tr_loss / global_step)
            fin_result = self.evaluate("validate")  # Only test set available for NSMC

            self.save_model(epoch)

        with open(os.path.join(self.args.result_dir, self.args.train_id, 'param_seach.txt'), "a", encoding="utf-8") as f:
            f.write('alpha: {}, gamma: {}, f1_macro: {}\n'.format(alpha, gamma, fin_result['f1_macro']))
        return fin_result['f1_macro']

    def evaluate(self, mode='test'):
        if mode == 'test':
            dataloader = self.test_dataloader
        elif mode == 'validate':
            dataloader = self.validate_dataloader
        else:
            raise Exception("Only dev and test dataset available")

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataloader) * self.args.batch_size)
        logger.info("  Batch size = %d", self.args.batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        self.model.eval()

        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'attention_mask': batch[1], 'labels': batch[3],
                          'token_type_ids': batch[2]}
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs['labels'].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(
                    out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        preds = np.argmax(preds, axis=1)
        result = compute_metrics(preds, out_label_ids)
        results.update(result)

        p_macro, r_macro, f_macro, support_macro \
            = precision_recall_fscore_support(y_true=out_label_ids, y_pred=preds,
                                              labels=[i for i in range(self.num_labels)], average='macro')

        results.update({
            'precision': p_macro, 'recall': r_macro, 'f1_macro': f_macro
        })

        with open(self.args.prediction_file, "w", encoding="utf-8") as f:
            for pred in preds:
                f.write("{}\n".format(pred))

        if mode == 'validate':
            logger.info("***** Eval results *****")
            for key in sorted(results.keys()):
                logger.info("  %s = %s", key, str(results[key]))

        return results

    def save_model(self, num=0):
        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(state, os.path.join(self.args.result_dir, self.args.train_id, 'epoch_' + str(num) + '.pth'))
        logger.info('model saved')

    def load_model(self, model_name):

        state = torch.load(os.path.join(self.args.result_dir, self.args.train_id, model_name))
        self.model.load_state_dict(state['model'])
        if self.optimizer is not None:
            self.optimizer.load_state_dict(state['optimizer'])
        if self.scheduler is not None:
            self.scheduler.load_state_dict(state['scheduler'])
        logger.info('model loaded')