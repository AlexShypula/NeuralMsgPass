import torch
import torch.nn.utils.rnn as rnn
import numpy as np
from dataloader_bert import get_datasets
import sys
import os
import pdb


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
MAX_SENTENCE_LENGTH = 512

batch_size = 8
num_task = 16

dict_er = {'apparel':12.5, 'baby':11.3, 'books':13.8, 'camera_photo':11.0, 'dvd':12.0, 'electronics':11.5,
            'health_personal_care':10.5, 'imdb':13.5, 'kitchen_housewares':11.8, 'magazines':7.8,
            'MR':22.0, 'music':14.3, 'software':12.8, 'sports_outdoors':13.3, 'toys_games':10.8, 'video':14.0, 'avg':12.7}

_, _, test_loader = get_datasets(train_batch_size = batch_size,
                                                     val_batch_size = batch_size,
                                                     test_batch_size = batch_size,
                                                     max_sentence_length = MAX_SENTENCE_LENGTH)

def test_one_batch(task_index, model, test_x, test_y, mask, task_type):
    with torch.no_grad():
        test_x = torch.stack(test_x, 0)
        test_x = test_x.to(DEVICE)
        test_y = test_y.to(DEVICE)
        mask = torch.stack(mask, 0)
        mask = mask.to(DEVICE)

        outputs = model(input_ids=test_x,
                        token_type_ids=None, 
                        attention_mask=mask,
                        labels=test_y)
        logits = outputs[1]

        # loss = criterion(outputs.view(-1),train_y.view(-1))
        binary_outputs = np.argmax(logits.data.cpu().numpy(),axis=1)

        return sum(binary_outputs == test_y.data.cpu().numpy().astype(np.int64))

result_file = open('test_result.txt', 'w')
cur_task = ""
test_corrects, test_count = 0, 0
avg = 0
path = os.path.join("bert","model.pt")
if not os.path.exists(path):
    print(path + " does not exist")
    exit(1)
print('...........Loading model ' + path)
model = torch.load(path)
model.to(DEVICE)
model.eval()
for test_batch_id, (test_x, test_y, mask, task_type) in enumerate(test_loader):
    if cur_task != task_type:
        if test_batch_id != 0:
            cur_test_acc = test_corrects / test_count
            avg += cur_test_acc
            print(cur_task + " accuracy: {:.3f}".format(cur_test_acc))
            result_file.write(cur_task + " accuracy: {:.3f}, ER: {:.3f}, paper ER: {:.3f}, diff: {:.3f}\n"
                    .format(cur_test_acc * 100, (1 - cur_test_acc) * 100, dict_er[cur_task],
                    (1 - cur_test_acc) * 100 - dict_er[cur_task]))
        cur_task = task_type
        test_corrects, test_count = 0, 0
        print('*** test begin ***')
    test_corrects += test_one_batch(test_loader.dataset.task2idx[task_type], model, test_x, test_y, mask, task_type)
    test_count += len(test_x)
cur_test_acc = test_corrects / test_count
avg += cur_test_acc
avg /= (len(dict_er) - 1)
print(cur_task + " accuracy: {:.3f}".format(cur_test_acc))
result_file.write(cur_task + " accuracy: {:.3f}, ER: {:.3f}, paper ER: {:.3f}, diff: {:.3f}\n\n"
        .format(cur_test_acc * 100, (1 - cur_test_acc) * 100, dict_er[task_type],
        (1 - cur_test_acc) * 100 - dict_er[task_type]))

result_file.write("average accuracy: {:.3f}, ER: {:.3f}, paper ER: {:.3f}, diff: {:.3f}\n"
        .format(avg * 100, (1 - avg) * 100, dict_er['avg'],
        (1 - avg) * 100 - dict_er['avg']))
result_file.close()