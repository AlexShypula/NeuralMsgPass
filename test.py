import torch
import torch.nn.utils.rnn as rnn
import numpy as np
from dataloader_v2 import get_datasets
import sys
import os
import pdb


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SENTENCE_LENGTH = 20000

batch_size = 8
num_task = 16

DATASETS = ['apparel', 'baby', 'books', 'camera_photo', 'dvd', 'electronics',
            'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines',
            'MR', 'music', 'software', 'sports_outdoors', 'toys_games', 'video', ]

_, _, test_loader = get_datasets(train_batch_size = batch_size,
                                                     val_batch_size = batch_size,
                                                     test_batch_size = batch_size,
                                                     max_sentence_length = MAX_SENTENCE_LENGTH)

def test_one_batch(test_loader, model, test_x, test_y, task_type):
    with torch.no_grad():
        task_index = test_loader.dataset.task2idx[task_type]
        test_x = rnn.pad_sequence(test_x, batch_first=True, padding_value=0)
        test_x = test_x.to(DEVICE)
        test_y = test_y.to(DEVICE)

        mask = test_x != 0
        mask = mask.float()
        test_out = model(test_x, mask, task_index)
        binary_outputs = np.where(test_out.view(-1).data.cpu().numpy() > 0.5, 1, 0)
        return sum(binary_outputs == test_y.data.cpu().numpy().astype(np.int64))


cur_task = ""
test_corrects, test_count = 0, 0
for test_batch_id, (test_x, test_y, task_type) in enumerate(test_loader):
    if cur_task != task_type:
        if test_batch_id != 0:
            cur_test_acc = test_corrects / test_count
            print(cur_task + " accuracy: {:.3f}".format(cur_test_acc))
        cur_task = task_type
        path = "model_" + task_type + ".pt"
        if not os.path.exists(path):
            print(path + " does not exist")
            break
        print('...........Loading model ' + path)
        model = torch.load(path)
        model.to(DEVICE)
        model.eval()
        test_corrects, test_count = 0, 0
        print('*** test begin ***')
    test_corrects += test_one_batch(test_loader, model, test_x, test_y, task_type)
    test_count += test_x.size(0)