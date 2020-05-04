import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.optim as optim
import time
import numpy as np
import sys
import os
from transformers import BertForSequenceClassification, BertConfig, AdamW, get_linear_schedule_with_warmup
from dataloader_bert import get_datasets
from bert import BertForNLI
import pdb
from collections import defaultdict
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SENTENCE_LENGTH = 512

batch_size = 16
num_epoch = 5

train_loader, val_loader = get_datasets(train_batch_size = batch_size,
                                                     val_batch_size = batch_size,
                                                     test_batch_size = batch_size,
                                                     max_sentence_length = MAX_SENTENCE_LENGTH)

bert_type = 'bert-base-uncased'
path = "bert.pt"
config = BertConfig.from_pretrained(bert_type)
label2idx = train_loader.dataset.label2idx
if os.path.exists(path):
    model = torch.load(path)
else:
    model = BertForNLI.from_pretrained(bert_type, label2idx=label2idx)
    # model = BertForSequenceClassification.from_pretrained(bert_type, num_labels=len(label2idx["MNLI"]))
model.resize_token_embeddings(len(train_loader.iterator.dataset.tokenizer))
model.to(DEVICE)
total_steps = len(train_loader) * num_epoch
optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, # Default value in run_glue.py
                                            num_training_steps = total_steps)


# criterion = nn.BCELoss(reduction='mean')
# criterion = criterion.to(DEVICE)

cur_valid_acc, best_valid_acc = 0.0, 0.0
train_acc = defaultdict(int)
train_total = defaultdict(int)
val_acc = defaultdict(int)
val_total = defaultdict(int)
for e in range(num_epoch):
    before = time.time()
    corrects = 0
    cum_loss = 0
    total = 0
    for batch_id, (train_x, train_y, mask, task_type) in enumerate(train_loader):
        # task_type = 'MNLI'
        model.train()
        # train_x = rnn.pad_sequence(train_x, batch_first=True, padding_value=0)
        train_x = torch.stack(train_x, 0)
        train_x = train_x.to(DEVICE)
        train_y = train_y.to(DEVICE)
        mask = torch.stack(mask,0)
        mask = mask.to(DEVICE)
        optimizer.zero_grad()

        # mask = train_x != 0
        # mask = mask.float()
        outputs = model(input_ids=train_x,
                        token_type_ids=None, 
                        attention_mask=mask,
                        labels=train_y,
                        task=task_type)
        loss = outputs[0]
        logits = outputs[1]

        # loss = criterion(outputs.view(-1),train_y.view(-1))
        binary_outputs = np.argmax(logits.data.cpu().numpy(),axis=1)

        correct = sum(binary_outputs == train_y.data.cpu().numpy().astype(np.int64))
        corrects += correct
        train_acc[task_type] += correct
        train_total[task_type] += train_x.size(0)
        total += train_x.size(0)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        cum_loss += loss.detach().item()
        optimizer.step()
        # Update the learning rate.
        scheduler.step()
        if batch_id != 0 and batch_id % 50 == 0:
            after = time.time()
            cum_loss = cum_loss / 50
            print("Batch [{}/{}], loss: {:.3f}, accuracy: {:.3f}, Time elapsed: {}".format(
                batch_id, len(train_loader), cum_loss, corrects / total, after - before))
            print(["{} accuracy: {:.3f}".format(k, v/train_total[k]) for k,v in train_acc.items()])
            sys.stdout.flush()
            cum_loss = 0
            before = after
            total = 0
            corrects = 0
        del train_x
        del train_y

        # if batch_id != 0 and batch_id % 1000 == 0:
        # validation after each epoch
    with torch.no_grad():
        model.eval()
        val_correct, val_corrects, val_total_loss, val_count = 0, 0, 0, 0
        print('*** epoch: {} validation begin ***'.format(e))
        val_begin = time.time()
        for val_batch_id, (val_x, val_y, mask, task_type) in enumerate(val_loader):
            # sometime the val_loader will somehow return empty val_x
            # which will break pad_sequence
            val_x = torch.stack(val_x, 0)
            val_x = val_x.to(DEVICE)
            val_y = val_y.to(DEVICE)
            mask = torch.stack(mask, 0)
            mask = mask.to(DEVICE)

            # mask = val_x != 0
            # mask = mask.long()
            outputs = model(input_ids=val_x,
                            attention_mask=mask,
                            labels=val_y,
                            task=task_type)
            val_loss = outputs[0]
            logits = outputs[1]
            val_total_loss += val_loss.detach().item()
            binary_outputs = np.argmax(logits.data.cpu().numpy(),axis=1)

            val_correct = sum(binary_outputs == val_y.data.cpu().numpy().astype(np.int64))
            val_corrects += correct
            val_count += val_x.size(0)
            val_acc[task_type] += correct
            val_total[task_type] += val_x.size(0)
        val_end = time.time()
        cur_valid_acc = val_corrects / val_count
        if cur_valid_acc > best_valid_acc:
            best_valid_acc = cur_valid_acc
        print('@@@ Saving Model with Accuracy: {}% @@@\n'.format(cur_valid_acc))
        torch.save(model, 'model{}_{:3f}.pt'.format(e,cur_valid_acc))
        print("loss: {:.3f}, accuracy: {:.3f}, Time elapsed: {} s".format(val_total_loss / val_count,
                                                                    cur_valid_acc,
                                                                    val_end - val_begin))
        print(["{} accuracy: {:.3f}".format(k, v/val_acc[k]) for k,v in val_acc.items()])
