import re
import os
import logging

import torch

import emoji
from soynlp.normalizer import repeat_normalize
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from config import Config

logger = logging.getLogger(__name__)


class MyDataset(Dataset):
    def __init__(self, mode):
        lines, labels = self.get_examples(mode)
        self.tokenizer = AutoTokenizer.from_pretrained(Config.BERT_MODEL_NAME)
        encoding = self.tokenizer(lines, return_tensors='pt', padding=True, truncation=True)

        self.input_ids = torch.LongTensor(encoding['input_ids'])
        self.attention_mask = torch.LongTensor(encoding['attention_mask'])
        self.labels = torch.LongTensor(labels)
        self.token_type_ids = torch.LongTensor(encoding['token_type_ids'])

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())

            return lines

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        emojis = ''.join(emoji.UNICODE_EMOJI.keys())
        p0 = re.compile(f'[^ .,?!/@$%~％·∼()\x00-\x7Fㄱ-힣{emojis}]+')
        p1 = re.compile(r'(\"){2,}')
        p2 = re.compile(r'(\?){2,}')
        p3 = re.compile(r'(\!){2,}')
        p4 = re.compile(r'(\.){2,}')
        p5 = re.compile(r'(\~){2,}')
        p6 = re.compile(r'(\;){2,}')
        p7 = re.compile(r'(\^){2,}')
        p8 = re.compile(r'(\*){2,}')
        p9 = re.compile(
            r'https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)')

        pattern = [(' ', p0), ("", p1), ("?", p2), ("!", p3), ("...", p4), ("~", p5), (";", p6), ("^^", p7), ("*", p8),
                   (" ", p9)]

        def clean(x):
            for ch, sub_pattern in pattern:
                x = sub_pattern.sub(ch, x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        text_a = []
        labels = []
        for i in range(1, len(lines)):
            line = lines[i].split('\t')

            if len(line) != 2:
                continue

            text_a.append(clean(line[0]))
            labels.append(int(line[1]))
        return text_a, labels

    def get_examples(self, mode):
        """
        Args:
            mode: train, dev, test
        """
        file_to_read = None
        if mode == 'train':
            file_to_read = mode + '.tsv'
        elif mode == 'validate':
            file_to_read = mode + '.tsv'
        elif mode == 'test':
            file_to_read = mode + '.tsv'

        logger.info("LOOKING AT {}".format(os.path.join(Config.DATASET_PATH, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(Config.DATASET_PATH, file_to_read)), mode)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {'inputs': self.input_ids[idx], 'mask': self.attention_mask[idx], 'labels': self.labels[idx], 'token_ids': self.token_type_ids}
