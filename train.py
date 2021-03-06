import pickle
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.optim as optim
import time
import numpy as np
from modules.ic_model import my_model
from dataloader_v2 import get_datasets
import sys
import os
import pdb


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SENTENCE_LENGTH = 600

batch_size = 8
num_task = 16
embedding_dim = 300
hidden_size = 200
message_size = 2 * hidden_size

#train_loader, val_loader, test_loader = load_data(batch_size=batch_size)
train_loader, val_loader, test_loader = get_datasets(train_batch_size = batch_size,
                                                     val_batch_size = batch_size,
                                                     test_batch_size = batch_size,
                                                     max_sentence_length = MAX_SENTENCE_LENGTH)

with open('processed-dataset/embedding.pkl', 'rb') as f:
    pre_trained_embedding = pickle.load(f)

path = "model.pt"
if os.path.exists(path):
    model = torch.load(path)
else:
    model = my_model(pre_trained_embedding, embedding_dim, num_task, 64250, hidden_size, message_size)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)

num_epoch = 10

criterion = nn.BCELoss(reduction='mean')
criterion = criterion.to(DEVICE)

cur_valid_acc, best_valid_acc = 0.0, 0.0

for e in range(num_epoch):
    before = time.time()
    corrects = 0
    cum_loss = 0
    total = 0
    for batch_id, (train_x, train_y, task_type) in enumerate(train_loader):
        model.train()
        task_index = train_loader.dataset.task2idx[task_type]
        train_x = rnn.pad_sequence(train_x, batch_first=True, padding_value=0)
        train_x = train_x.to(DEVICE)
        train_y = train_y.to(DEVICE)

        mask = train_x != 0
        mask = mask.float()
        outputs = model(train_x, mask, task_index)
        assert all(outputs >0)
        assert all(outputs < 1)
        assert all(train_y >= 0)
        assert all(train_y <= 1)
        loss = criterion(outputs.view(-1),train_y.view(-1))
        binary_outputs = np.where(outputs.view(-1).data.cpu().numpy() > 0.5, 1, 0)

        correct = sum(binary_outputs == train_y.data.cpu().numpy().astype(np.int64))
        corrects += correct
        total += train_x.size(0)

        optimizer.zero_grad()
        loss.backward()
        cum_loss += loss.detach().item()
        optimizer.step()
        if batch_id != 0 and batch_id % 50 == 0:
            after = time.time()
            cum_loss = cum_loss / 50
            print("Batch [{}/{}], loss: {:.3f}, accuracy: {:.3f}, Time elapsed: {}".format(
                batch_id, len(train_loader), cum_loss, corrects / total, after - before))
            sys.stdout.flush()
            cum_loss = 0
            before = after
            total = 0
            corrects = 0
        del train_x
        del train_y

        if batch_id != 0 and batch_id % 900 == 0:
        # validation after each epoch
            with torch.no_grad():
                model.eval()
                val_correct, val_corrects, val_total_loss, val_count = 0, 0, 0, 0
                print('*** epoch: {} validation begin ***'.format(e))
                val_begin = time.time()
                for val_batch_id, (val_x, val_y, task_type) in enumerate(val_loader):

                    # sometime the val_loader will somehow return empty val_x
                    # which will break pad_sequence
                    task_index = train_loader.dataset.task2idx[task_type]
                    if len(val_x) == 0:
                        print('empty val_x')
                        continue
                    val_x = rnn.pad_sequence(val_x, batch_first=True, padding_value=0)
                    val_x = val_x.to(DEVICE)
                    val_y = val_y.to(DEVICE)

                    mask = val_x != 0
                    mask = mask.float()
                    val_out = model(val_x, mask, task_index)
                    val_loss = criterion(val_out.view(-1), val_y.view(-1))
                    val_total_loss += val_loss.detach().item()
                    binary_outputs = np.where(val_out.view(-1).data.cpu().numpy() > 0.5, 1, 0)
                    val_correct = sum(binary_outputs == val_y.data.cpu().numpy().astype(np.int64))
                    val_corrects += val_correct
                    val_count += val_x.size(0)
                val_end = time.time()
                cur_valid_acc = val_corrects / val_count
                if cur_valid_acc > best_valid_acc:
                    best_valid_acc = cur_valid_acc
                    print('@@@ Saving Best Model with Accuracy: {}% @@@\n'.format(cur_valid_acc))
                    torch.save(model, 'model.pt')
                print("loss: {:.3f}, accuracy: {:.3f}, Time elapsed: {} s".format(val_total_loss / val_count,
                                                                          cur_valid_acc,
                                                                          val_end - val_begin))
