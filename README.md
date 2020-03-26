# NeuralMsgPass

Dataloader:

Call load_data(batch_size) will return train_loader, val_loader, test_loader

Iterate through the loader outputs 3 things:
1. A list of sorted word index pytorch tensors(float32). List length == batch_size
2. A list of labels(int) either 0 or 1. List length = batch_size
3. task_type: a string that indicate which task this batch belongs to

Usage:

for batch in xxx_loader:\\
	word_idxs, labels, task_type = batch\\
	pad and pack sequence maybe\\
	to device\\
	...

Pretrained embedding:

The vocab size of this dataset is 64250 (derived from train and val dataset). I uploaded pretrained embedding to our googld drive https://drive.google.com/drive/u/1/folders/107SX_vOw1-NKtU-YlAonIy5FE80GE0HL\\
pretrained embedding numpy array shape is [64250, 300],(from glove.840B.300d.txt) Download and put it under processed-dataset/embedding.pkl\\
load it in this way:\\
import pickle
with open('processed-dataset/embedding.pkl', 'rb') as f:\\
    w_emb = pickle.load(f)\\
