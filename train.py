import pickle
import torch
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
import time
from model import my_model
from dataloader import load_data


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 64
num_task = 16
embedding_dim = 300
hidden_size = 200
message_size = 100
train_loader, val_loader, test_loader = load_data(batch_size=batch_size)

with open('processed-dataset/embedding.pkl', 'rb') as f:
    pre_trained_embedding = pickle.load(f)

model = my_model(pre_trained_embedding, embedding_dim, num_task, hidden_size, message_size)
model.to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

num_epoch = 10

for e in range(num_epoch):
    model.train()
    start_time = time.time()
    for train_x, train_y, task_type in train_loader:
        preds = model(train_x)

