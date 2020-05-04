import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import bisect
import time
import pdb
from typing import List
import random
import os
import math
import codecs
import csv
import sys
from transformers import BertTokenizer

csv.field_size_limit(sys.maxsize//10)


random.seed(88888888)

PATH_TO_DATA = "glue_data"
MODE_TRAIN = "train"
MODE_VAL = "dev"
MODE_TEST = "test"
DATASETS = ['CoLA', 'MNLI', 'MRPC', 'QNLI', 'QQP', 'RTE',
            'SNLI', 'SST-2', 'STS-B', 'WNLI']

NLI = ['MNLI']


class sentimentDataset(Dataset):

    def lengths2indices(self, task_lengths: List[int]):
        return [0]+list(np.cumsum(task_lengths))[:-1]

    def __init__(self, path_to_data: str, dataset_names: List[str], mode: str, sentence_length_threshold = 350):  # mode is [train, val, test]
        super(sentimentDataset, self).__init__()

        self.label2idx = {d: {} for d in dataset_names}
        self.idx2label = {d: {} for d in dataset_names}
        self.sentence_tensors = []
        self.labels = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.attention_masks = []
        for i, dataset_name in enumerate(dataset_names):
            file_name = os.path.join(path_to_data, dataset_name, mode + '.tsv')
            if dataset_name == 'MNLI':
                if mode == 'dev':
                    file_name = os.path.join(path_to_data, dataset_name, 'dev_matched.tsv')
                if mode == 'test':
                    file_name = os.path.join(path_to_data, dataset_name, 'test_matched.tsv')
            with open(file_name) as f:
                rd = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
                n_added = 0
                label_idx = 0
                for line in rd:
                    l = 'gold_label'
                    label = line[l]
                    if label not in self.label2idx[dataset_name]:
                        self.label2idx[dataset_name][label] = label_idx
                        self.idx2label[dataset_name][label_idx] = label
                        label_idx += 1
                    label = self.label2idx[dataset_name][label]
                    self.labels.append(label)
                    sentence1 = self.tokenizer.tokenize(line['sentence1'])
                    sentence2 = self.tokenizer.tokenize(line['sentence2'])
                    # sentence = ['<tsk{}>'.format(i)] + sentence1 + [self.tokenizer.sep_token] + sentence2 + [self.tokenizer.sep_token]
                    # sentence = [self.tokenizer.cls_token] + sentence1 + [self.tokenizer.sep_token] + sentence2
                    encoded_dict = self.tokenizer.encode_plus(
                        text=sentence1,                      # Sentence to encode.
                        text_pair=sentence2,
                        add_special_tokens = True, # don't add '[CLS]' and '[SEP]'
                        max_length = sentence_length_threshold,           # Pad & truncate all sentences.
                        pad_to_max_length = True,
                        return_attention_mask = True,   # Construct attn. masks.
                        return_tensors = 'pt',     # Return pytorch tensors.
                    )
                    n_added+=1
    
                    # Add the encoded sentence to the list.    
                    self.sentence_tensors.append(encoded_dict['input_ids'].view(-1))
                    # And its attention mask (simply differentiates padding from non-padding).
                    self.attention_masks.append(encoded_dict['attention_mask'].view(-1))

                    # dataset is super big, debug with small size:
                    if n_added == 10000:
                        break


    def __getitem__(self, i):
        return self.sentence_tensors[i], self.labels[i], self.attention_masks[i], NLI[0]

    def __len__(self):
        return len(self.sentence_tensors)


def get_datasets(train_batch_size, val_batch_size, test_batch_size, max_sentence_length = 350):

    _train_dataset = sentimentDataset(PATH_TO_DATA, NLI, MODE_TRAIN, max_sentence_length)
    _val_dataset = sentimentDataset(PATH_TO_DATA, NLI, MODE_VAL, max_sentence_length)
    # _test_dataset = sentimentDataset(PATH_TO_DATA, NLI, MODE_TEST, max_sentence_length)

    _train_loader = DataLoader(_train_dataset, batch_size=train_batch_size, shuffle=True)
    _val_loader = DataLoader(_val_dataset, batch_size=val_batch_size, shuffle= False)
    # _test_loader = sentimentDataLoader(_test_dataset, test_batch_size, shuffle = False)

    # return _train_loader, _val_loader, _test_loader
    return _train_loader, _val_loader
