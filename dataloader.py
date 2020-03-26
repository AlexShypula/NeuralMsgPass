import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import bisect
import time
import pdb

datasets = ['apparel','baby','books','camera_photo','dvd','electronics',
            'health_personal_care','imdb','kitchen_housewares','magazines',
            'MR','music','software','sports_outdoors','toys_games','video',]
class sentimentDataset(Dataset):
    
    def __init__(self,mode, batch_size): # mode is [train, val, test]
        self.mode = mode
        self.batch_size = batch_size
        self.ids_list = [] # 16 lists of sentence ids
        self.labels_list = []
        self.lengths = []
        self.lengths_incr = []
        for dataset in datasets:
            ids = []
            labels = []
            l = 0
            with open('processed-dataset/' + dataset + '.' + mode) as f:
                for line in f:
                    l += 1
                    parsed = line.strip().split('\t')
                    labels.append(int(parsed[0]))
                    ids.append(torch.tensor(np.array([float(x) for x in parsed[1].split()]), dtype=torch.float32))
            self.lengths.append(l)
            self.ids_list.append(ids)
            self.labels_list.append(labels)
        prev = 0
        for x in self.lengths:
            cur = prev + x
            self.lengths_incr.append(cur)
            prev = cur

    def __getitem__(self, i):
        start = i * self.batch_size
        dataset_idx = bisect.bisect_right(self.lengths_incr, start)
        start = start - self.lengths_incr[dataset_idx]
        task_type = datasets[dataset_idx]
        ids = self.ids_list[dataset_idx]
        labels= self.labels_list[dataset_idx]
        return ids[start: min(len(ids), start + self.batch_size)], \
                labels[start: min(len(ids), start + self.batch_size)], \
                task_type

    def __len__(self):
        return int(sum(self.lengths) / self.batch_size)

# collate fn lets you control the return value of each batch
# for packed_seqs, you want to return your data sorted by length
def collate_lines(seq_list):
    inputs, targets, task_type = seq_list[0]
    lens = [len(seq) for seq in inputs]
    seq_order = sorted(range(len(lens)), key=lens.__getitem__, reverse=True)
    inputs = [inputs[i] for i in seq_order]
    targets= [targets[i] for i in seq_order]
    return inputs, targets, task_type

def load_data(batch_size):

    train_dataset = sentimentDataset("train", batch_size)
    val_dataset = sentimentDataset("val", batch_size)
    test_dataset = sentimentDataset("test", batch_size)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, collate_fn = collate_lines)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=1, collate_fn = collate_lines)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=1, collate_fn = collate_lines)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    x, _, _ = load_data(8)
    for i in x:
        _, _ , task = i
        pdb.set_trace()
        print(task)