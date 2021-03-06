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
from transformers import BertTokenizer

random.seed(88888888)

PATH_TO_DATA = "mtl-dataset"
MODE_TRAIN = "train"
MODE_VAL = "val"
MODE_TEST = "task.test"
DATASETS = ['apparel', 'baby', 'books', 'camera_photo', 'dvd', 'electronics',
            'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines',
            'MR', 'music', 'software', 'sports_outdoors', 'toys_games', 'video', ]


class sentimentDataset(Dataset):

    def lengths2indices(self, task_lengths: List[int]):
        return [0]+list(np.cumsum(task_lengths))[:-1]

    def __init__(self, path_to_data: str, dataset_names: List[str], mode: str, sentence_length_threshold = 350):  # mode is [train, val, test]
        super(sentimentDataset, self).__init__()

        self.task2idx = {task: i for i, task in enumerate(dataset_names)}
        self.idx2task = {i: task for i, task in enumerate(dataset_names)}

        self.mode = mode
        self.sentence_tensors = []
        self.labels = []
        self.task_lengths = []
        self.task_beginning_indices = []
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.attention_masks = []
        for dataset_name in dataset_names:
            with open(os.path.join(path_to_data, dataset_name) + '.' + mode, \
                encoding='utf-8', errors='ignore') as f:
                n_added = 0
                for i, line in enumerate(f):
                    line = line.strip()
                    label = line[0]
                    sentence = line[2:]
                    self.labels.append(int(label)) #TODO typecheck
                    encoded_dict = self.tokenizer.encode_plus(
                        sentence,                      # Sentence to encode.
                        add_special_tokens = True, # Add '[CLS]' and '[SEP]'
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
                    # # tokenized_review = self.tokenizer.tokenize(sentence)
                    # if len(ids_review) <= sentence_length_threshold:
                    #     self.labels.append(int(label)) #TODO typecheck
                    #     # ids_review  = self.tokenizer.convert_tokens_to_ids(tokenized_review)
                    #     ids_review = ids_review + [0] * (sentence_length_threshold - len(ids_review))
                    #     sentence_tensor = torch.tensor(ids_review,dtype=torch.long)
                    #     self.sentence_tensors.append(sentence_tensor)
                    #     n_added+=1

            self.task_lengths.append(n_added)
        self.task_beginning_indices = self.lengths2indices(self.task_lengths)

    def __getitem__(self, i):
        task_index = bisect.bisect_right(self.task_beginning_indices, i) - 1
        return self.sentence_tensors[i], self.labels[i], self.attention_masks[i], self.idx2task[task_index]

    def shuffle(self):
        index_list = self.task_beginning_indices + [self.__len__()]
        new_sentence_tensors = []
        new_labels = []
        news_attention_masks = []
        for task_index in range(len(index_list)-1):
            beginning_index = index_list[task_index]
            end_index = index_list[task_index+1]
            sentences = self.sentence_tensors[beginning_index: end_index]
            masks = self.attention_masks[beginning_index: end_index]
            labels = self.labels[beginning_index: end_index]
            rand_index = list(range(len(sentences)))
            random.shuffle(rand_index)
            new_sentence_tensors.extend([sentences[i] for i in rand_index])
            new_labels.extend([labels[i] for i in rand_index])
            news_attention_masks.extend([masks[i] for i in rand_index])

        self.sentence_tensors = new_sentence_tensors
        self.labels = new_labels
        self.attention_masks = news_attention_masks

    def __len__(self):
        return len(self.sentence_tensors)


class batch_iterator:
    def __init__(self, dataset: Dataset, batch_size: int, shuffle: bool):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.number_batches = 0

        self.task_first_batch_index = []
        self.task_last_batch_indices = []

        for l in dataset.task_lengths:
            self.task_first_batch_index.append(self.number_batches)
            task_batches = math.ceil(l / batch_size)
            self.number_batches += task_batches
            self.task_last_batch_indices.append(self.number_batches-1)


    def get_index_info(self, index):

        task_index = bisect.bisect_left(self.task_last_batch_indices, index)
        batch_offset = index - self.task_first_batch_index[task_index]
        dataset_index_offset = self.batch_size * batch_offset
        dataset_start_idx = dataset_index_offset + self.dataset.task_beginning_indices[task_index]

        if index != self.task_last_batch_indices[task_index] or (self.dataset.task_lengths[task_index] % self.batch_size) == 0:
            return self.batch_size, task_index, dataset_start_idx
        else:
            return self.dataset.task_lengths[task_index] % self.batch_size, task_index, dataset_start_idx

    def __call__(self, task_index):

        beginning_index = self.task_first_batch_index[task_index]
        last_index = self.task_last_batch_indices[task_index]
        for i in range(beginning_index, last_index + 1): # add 1 for inclusive iteration
            functional_batch_size, batch_task_index, dataset_start_idx = self.get_index_info(i)
            try:
                assert batch_task_index == task_index
            except AssertionError as error:
                print(f"error: {error}\nbatch_task_index is {batch_task_index} and task index is {task_index}")
                print(f"batch task is {self.dataset.task2idx[batch_task_index]} and task is {self.dataset.task2idx[task_index]}")
            sentences = []
            labels = []
            for j in range(functional_batch_size):
                sentence, label, task = self.dataset[dataset_start_idx + j]
                assert task == self.dataset.idx2task[task_index]
                sentences.append(sentence)
                labels.append(label)
            # print(f"sentences are {sentences} and labels are {labels} and task is {task}")
            yield sentences, torch.tensor(labels, dtype=torch.long), task


    def __iter__(self):
        index_queue = list(range(self.number_batches))

        if self.shuffle:
            random.shuffle(index_queue)
            self.dataset.shuffle()

        for i in index_queue:
            functional_batch_size, task_index, dataset_start_idx = self.get_index_info(i)
            sentences = []
            labels = []
            masks = []
        
            #print(f"i is {i} and functional batch size is {functional_batch_size} task index is {task_index} dataset start idx is {dataset_start_idx}")
            for j in range(functional_batch_size):
                sentence, label, mask, task = self.dataset[dataset_start_idx+j]
                assert task == self.dataset.idx2task[task_index]
                sentences.append(sentence)
                labels.append(label)
                masks.append(mask)
            #print(f"sentences are {sentences} and labels are {labels} and task is {task}")
            yield sentences, torch.tensor(labels, dtype = torch.long), masks, task

    def __len__(self):
        return self.number_batches



class sentimentDataLoader(DataLoader):
    """
    """
    def __init__(self, dataset, batch_size, shuffle=True):
        super(sentimentDataLoader, self).__init__(dataset, batch_size = batch_size)
        # note that super sets dataset to self.dataset and batchsize to self.batch_size

        self.shuffle = shuffle
        self.iterator = batch_iterator(self.dataset, self.batch_size, self.shuffle)

    def __call__(self, task_index):
        yield from self.iterator(task_index)

    def __iter__(self):
        # concatenate your articles and build into batches
        yield from self.iterator

    def __len__(self):
        # gives the number of batches in the iterator
        return len(self.iterator)

def get_datasets(train_batch_size, val_batch_size, test_batch_size, max_sentence_length = 350):

    _train_dataset = sentimentDataset(PATH_TO_DATA, DATASETS, MODE_TRAIN, max_sentence_length)
    _val_dataset = sentimentDataset(PATH_TO_DATA, DATASETS, MODE_VAL, max_sentence_length)
    _test_dataset = sentimentDataset(PATH_TO_DATA, DATASETS, MODE_VAL, max_sentence_length)

    _train_loader = sentimentDataLoader(_train_dataset, train_batch_size, shuffle = True)
    _val_loader = sentimentDataLoader(_val_dataset, val_batch_size, shuffle= False)
    _test_loader = sentimentDataLoader(_test_dataset, test_batch_size, shuffle = False)

    return _train_loader, _val_loader, _test_loader


if __name__ == "__main__":
    dataset = sentimentDataset("tmp", ['single', 'tens', 'twenties'], "demo")
    print(f"dataset sentence tensors are {dataset.sentence_tensors} and the corresponding labels are dataset {dataset.labels}")
    dataloader = sentimentDataLoader(dataset, batch_size=3, shuffle=True)
    print(f"batch size is set to 3 and we will do three epochs, note that shuffle is set to {True}")
    for epoch in range(3):
        print(f"epoch {epoch}: ")
        for i, (sentence_tensor, labels, task) in enumerate(dataloader):
            print(f"batch number {i+1} is of task: {task}\nhas sentence tensor: {sentence_tensor}\nlabels: {labels}")