import pickle
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import time
from modules.model import my_model
from dataloader import load_data
import pdb

DATASETS = ['apparel', 'baby', 'books', 'camera_photo', 'dvd', 'electronics',
            'health_personal_care', 'imdb', 'kitchen_housewares', 'magazines',
            'MR', 'music', 'software', 'sports_outdoors', 'toys_games', 'video']

task_to_idx = {}
for idx, task in enumerate(DATASETS):
    task_to_idx[task] = idx

# pdb.set_trace()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
num_task = 16
embedding_dim = 200
hidden_size = 200
message_size = hidden_size
train_loader, val_loader, test_loader = load_data(batch_size=batch_size)

with open('processed-dataset/embedding.pkl', 'rb') as f:
    pre_trained_embedding = pickle.load(f)

model = my_model(pre_trained_embedding, embedding_dim, num_task, 64250, hidden_size, message_size)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

num_epoch = 10
# pdb.set_trace()
for e in range(num_epoch):
    model.train()
    start_time = time.time()
    for train_x, train_y, task_type in train_loader:
        # train_x.to(DEVICE)
        # train_y.to(DEVICE)
        preds = model(train_x, task_to_idx[task_type])
        pdb.set_trace()

