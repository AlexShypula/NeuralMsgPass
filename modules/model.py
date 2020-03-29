import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
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
        # self.embed = nn.Embedding.from_pretrained(vocab)
        self.embed.weight.data.copy_(pretrained_embedding)
        self.share_lstm = nn.LSTM(self.embed_dim, self.hidden_size,
                                  num_layers=1, batch_first=self.batch_first, bidirectional=self.bidirectional)
        self.task_specific_lstm_list = [message_lstm(self.embed_dim,
                                                     self.hidden_size,
                                                     self.message_size,
                                                     self.bias,
                                                     self.batch_first,
                                                     self.bidirectional)] * self.num_tasks

        self.aggregation_activation = nn.Tanh()
        self.Ws = nn.Linear(self.embed_dim + 2 * self.hidden_size, self.hidden_size)
        self.Us = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Softmax(dim=1)
        self.fc = nn.Linear(self.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, mask, TASK):

        embeddings = self.embed(x)  # B x T x E

        shared_task_output, (h_shared, c_n) = self.share_lstm(embeddings)  # shared_task_output: T x B x H
        h_shared = h_shared.transpose(0, 1)  # batch first B x 1 x H
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
            norm_h_shared = B * shared_task_output # B x T x H
            # norm_h_shared = torch.mul(B, )  # (B x T x 1 ) x (B x T x H) -> (B x T x H)
            Rt = torch.sum(norm_h_shared, dim=1)  # (B x H)
            output, state_h, state_c = task_specific_lstm._step(embeddings[:, t, :], Rt, state_h, state_c)
            outputs.append(output)
            h_task = state_h.unsqueeze(1)  # B x 1 x H
            h_task = h_task.repeat(1, shared_task_output.size(1), 1)  # state_h was B x H -> B x T x H
            # pdb.set_trace()
        # out = torch.stack(outputs, axis=1)
        out = self.fc(output)
        out = self.sigmoid(out)
        return out