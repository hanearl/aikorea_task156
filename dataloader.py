import os
import copy
import json
import logging

import torch
from torch.utils import data
from torch.utils.data import TensorDataset
from torch.utils.data import Dataset

import re
import emoji
from soynlp.normalizer import repeat_normalize

from transformers import AutoTokenizer

from config import Config
from new_dataloader import MyDataset

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
        label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """

    def __init__(self, guid, text_a, label):
        self.guid = guid
        self.text_a = text_a
        self.label = label

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, label_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.label_id = label_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class NsmcProcessor(object):
    """Processor for the NSMC data set """

    def __init__(self, root):
        self.root = root

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

        pattern = [(' ', p0), ("", p1), ("?", p2), ("!", p3), ("...", p4), ("~", p5), (";", p6), ("^^", p7), ("*", p8), (" ", p9)]

        def clean(x):
            for ch, sub_pattern in pattern:
                x = sub_pattern.sub(ch, x)
            x = x.strip()
            x = repeat_normalize(x, num_repeats=2)
            return x

        examples = []
        # for (i, line) in (lines[1:]):
        for i in range(1, len(lines)):
            line = lines[i].split('\t')
            guid = "%s-%s" % (set_type, i)
            try:
                text_a = clean(line[0])
                label = int(line[1])
            except:
                print('[exept] ', line)
                continue
            # if i % 1000 == 0:
            #     logger.info(lines[i])
            examples.append(InputExample(guid=guid, text_a=text_a, label=label))
        return examples

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

        logger.info("LOOKING AT {}".format(os.path.join(self.root, file_to_read)))
        return self._create_examples(self._read_file(os.path.join(self.root, file_to_read)), mode)


processors = {
    "nsmc": NsmcProcessor,
}


def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = tokenizer(example.text_a, padding='max_length', truncation=True, max_length=max_seq_len, return_tensors="pt")

#         # Account for [CLS] and [SEP]
#         special_tokens_count = 2
#         if len(tokens) > max_seq_len - special_tokens_count:
#             tokens = tokens[:(max_seq_len - special_tokens_count)]

#         # Add [SEP] token
#         tokens += [sep_token]
#         token_type_ids = [sequence_a_segment_id] * len(tokens)

#         # Add [CLS] token
#         tokens = [cls_token] + tokens
#         token_type_ids = [cls_token_segment_id] + token_type_ids

#         input_ids = tokenizer.convert_tokens_to_ids(tokens)

#         # The mask has 1 for real tokens and 0 for padding tokens. Only real
#         # tokens are attended to.
#         attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        token_type_ids = tokens['token_type_ids']

        # assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        # assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
        #     len(attention_mask), max_seq_len)
        # assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
        #                                                                                           max_seq_len)

        label_id = example.label

        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label_id=label_id
                          ))

    return features


def load_and_cache_examples(root, tokenizer, mode):
    processor = processors['nsmc'](root)

    # Load data features from cache or dataset file
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        'nsmc', list(filter(None, Config.BERT_MODEL_NAME.split("/"))).pop(), 300, mode)

    cached_features_file = os.path.join(Config.TRAIN_DIR, cached_file_name)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", Config.TRAIN_DIR)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "validate":
            examples = processor.get_examples("validate")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(examples, 100, tokenizer)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_label_ids)
    return dataset


def data_loader(root, phase, batch_size, tokenizer):
    dataset = MyDataset(phase)

    if phase == 'train':
        sampler = data.RandomSampler(dataset)
    else:
        sampler = data.SequentialSampler(dataset)

    dataloader = data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)
    return dataloader