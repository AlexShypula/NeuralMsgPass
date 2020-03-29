import pickle
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.optim as optim
import time
import numpy as np
from modules.model import my_model
from dataloader import load_data
import sys

import pdb

DATASETS = ['apparel', 'baby', 'books', 'camera_photo', 'dvd', 'electronics',
            'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines',
            'MR', 'music', 'software', 'sports_outdoors', 'toys_games', 'video']

task_to_idx = {}
for idx, task in enumerate(DATASETS):
    task_to_idx[task] = idx

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 8
num_task = 16
embedding_dim = 300
hidden_size = 200
message_size = hidden_size
train_loader, val_loader, test_loader = load_data(batch_size=batch_size)

with open('processed-dataset/embedding.pkl', 'rb') as f:
    pre_trained_embedding = pickle.load(f)

model = my_model(pre_trained_embedding, embedding_dim, num_task, 64250, hidden_size, message_size)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

num_epoch = 10

criterion = nn.BCELoss(reduction = 'mean')
criterion = criterion.to(DEVICE)
for e in range(num_epoch):
    model.train()
    before = time.time()
    corrects = 0
    cum_loss = 0
    total = 0
    for batch_id, (train_x, train_y, task_type) in enumerate(train_loader):
        train_x = rnn.pad_sequence(train_x, batch_first = True, padding_value = 0)
        train_x = train_x.to(DEVICE)
        train_y = torch.tensor(train_y,dtype=torch.float32)
        train_y = train_y.to(DEVICE)

        mask = train_x != 0
        mask = mask.float()
        outputs = model(train_x, mask, task_to_idx[task_type])
        loss = criterion(outputs.view(-1),train_y.view(-1))
        binary_outputs = np.where(outputs.view(-1).data.cpu().numpy() > 0.5, 1, 0)
        
        correct = sum(binary_outputs == train_y.data.cpu().numpy().astype(np.int64))
        corrects += correct
        total += train_x.size(0)

        optimizer.zero_grad()
        loss.backward()
        cum_loss +=  loss.detach().item()
        optimizer.step()
        if batch_id % 50  == 0:
            after = time.time()
            cum_loss = cum_loss / 50
            print("Batch [{}/{}], loss: {:.3f}, accuracy: {:.3f}, Time elapsed: {}".format(\
                batch_id, len(train_loader), cum_loss, corrects / total, after - before))
            sys.stdout.flush()
            cum_loss = 0
            before = after
            total = 0
            corrects = 0
        del train_x
        del train_y