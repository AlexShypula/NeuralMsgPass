import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import torch.nn.functional as F
from .message_lstm import message_lstm
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
import pdb


class my_model(nn.Module):
    def __init__(self,
                 pretrained_embedding,
                 embed_dim: int,
                 num_tasks: int,
                 input_size: int,
                 hidden_size: int,
                 message_size: int = None,
                 bias: bool = True,
                 batch_first: bool = True,
                 bidirectional: bool = False):
        super(my_model, self).__init__()

        pretrained_embedding = torch.tensor(pretrained_embedding, dtype=torch.float64)
        self.num_tasks = num_tasks
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.message_size = message_size
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(self.input_size, self.embed_dim)
        self.embed.weight.data.copy_(pretrained_embedding)
        self.embed.weight.requires_grad=True
        self.share_lstm = nn.LSTM(self.embed_dim, self.hidden_size,
                                  num_layers=1, batch_first=self.batch_first, bidirectional=True)
        self.task_specific_lstm_list = nn.ModuleList()
        for _ in range(self.num_tasks):
            self.task_specific_lstm_list.append(message_lstm(self.embed_dim,
                                                     self.hidden_size,
                                                     self.message_size,
                                                     self.bias,
                                                     self.batch_first,
                                                     self.bidirectional))
        # self.task_specific_lstm_list = nn.ModuleList([message_lstm(self.embed_dim,
        #                                              self.hidden_size,
        #                                              self.message_size,
        #                                              self.bias,
        #                                              self.batch_first,
        #                                              self.bidirectional)] * self.num_tasks)
        self.sigmoid = nn.Sigmoid()
    
    def avgpool(self, x, mask):
        """do avg pool on time dimension
        
        Arguments:
            x {torch.FloatTensoro} -- hidden output from task specific lstm of shap B x T x H
            mask {[torch.FloatTensor]} -- B x T
        """
        summed = torch.sum(mask, 1) # B
        mask = mask.unsqueeze(2)
        x = x * mask # B T H
        x = torch.sum(x, 1) # B H
        x = x / summed.unsqueeze(1)
        return x

    def forward(self, x, mask, TASK):

        embeddings = self.embed(x)  # B x T x E

        shared_task_output, (_, _) = self.share_lstm(embeddings)  # shared_task_output: T x B x 2H
        h_task = torch.zeros(embeddings.size(0), embeddings.size(1), self.hidden_size).to(DEVICE)  # B x T x H
        state_h = torch.zeros(embeddings.size(0), self.hidden_size).to(DEVICE)  # B x H
        state_c = torch.zeros(embeddings.size(0), self.hidden_size).to(DEVICE)  # B x H

        task_specific_lstm = self.task_specific_lstm_list[TASK].to(DEVICE)
        outputs = []
        # loop for the "decoder" or task specific layer
        # pdb.set_trace()

        # loop through each time step
        for t in range(embeddings.size(1)):
            catted_input = torch.cat((shared_task_output, h_task, embeddings), dim=2)
            var1 = task_specific_lstm.Ws(catted_input)  # var 1 is B x T x H
            var2 = task_specific_lstm.aggregate_activation(var1)  # var 1 is B x T x H
            Si = task_specific_lstm.Us(var2)  # Si is B x T x 1
            B = task_specific_lstm.masked_softmax(Si, mask)  # B is B x T x 1, softmax over T
            B = B.unsqueeze(2)
            norm_h_shared = B * shared_task_output # B x T x 2H
            Rt = torch.sum(norm_h_shared, dim=1)  # (B x 2H)
            output, state_h, state_c = task_specific_lstm._step(embeddings[:, t, :], Rt, state_h, state_c)
            outputs.append(output)
            h_task = state_h.unsqueeze(1)  # B x 1 x H
            h_task = h_task.repeat(1, shared_task_output.size(1), 1)  # state_h was B x H -> B x T x H
        out = torch.stack(outputs, axis=1) # B x T x H
        # pdb.set_trace()
        out = self.avgpool(out, mask)
        out = task_specific_lstm.fc(out)
        out = self.sigmoid(out)
        return out