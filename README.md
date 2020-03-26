# NeuralMsgPass

Dataloader:

Call load_data(batch_size) will return train_loader, val_loader, test_loader

Iterate through the loader outputs 3 things:
1. A list of sorted word index pytorch tensors(float32). List length == batch_size
2. A list of labels(int) either 0 or 1. List length = batch_size
3. task_type: a string that indicate which task this batch belongs to

Usage:

for batch in xxx_loader:
	word_idxs, labels, task_type = batch
	pad and pack sequence maybe
	to device
	...
