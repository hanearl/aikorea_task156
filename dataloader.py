import os
import copy
import json
import logging

import torch
from torch.utils import data
from torch.utils.data import TensorDataset

from torchvision import datasets, transforms
from PIL import Image
import pandas as pd
from transformers import AutoTokenizer, AutoModel, AutoConfig

logger = logging.getLogger(__name__)

BERT_MODEL_NAME = 'beomi/kcbert-base'
MAX_SEQ_LEN = 100


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
        examples = []
        for i in range(1, len(lines)):
            line = lines[i].split('\t')
            guid = "%s-%s" % (set_type, i)
            if len(line) != 2:
                continue

            text_a = line[0]
            label = int(line[1])
            if i % 1000 == 0:
                logger.info(lines[i])
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


def convert_examples_to_features(examples, max_seq_len, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))

        tokens = tokenizer(example.text_a, padding='max_length', truncation=True, return_tensors='pt', max_length=max_seq_len)

        input_ids = tokens['input_ids'][0].tolist()
        attention_mask = tokens['attention_mask'][0].tolist()
        token_type_ids = tokens['token_type_ids'][0].tolist()

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(
            len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids),
                                                                                                  max_seq_len)

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

        break

    return features


def load_and_cache_examples(root, tokenizer, mode):
    processor = processors['nsmc'](root)

    # Load data features from cache or dataset file
    cached_file_name = 'cached_{}_{}_{}_{}'.format(
        'nsmc', list(filter(None, BERT_MODEL_NAME.split("/"))).pop(), MAX_SEQ_LEN, mode)

    cached_features_file = os.path.join(root, cached_file_name)
    if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", root)
        if mode == "train":
            examples = processor.get_examples("train")
        elif mode == "validate":
            examples = processor.get_examples("validate")
        elif mode == "test":
            examples = processor.get_examples("test")
        else:
            raise Exception("For mode, Only train, dev, test is available")

        features = convert_examples_to_features(examples, MAX_SEQ_LEN, tokenizer)
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
    dataset = load_and_cache_examples(root, tokenizer, mode=phase)

    if phase == 'train':
        sampler = data.RandomSampler(dataset)
    else:
        sampler = data.SequentialSampler(dataset)

    dataloader = data.DataLoader(dataset=dataset, sampler=sampler, batch_size=batch_size)
    return dataloader
