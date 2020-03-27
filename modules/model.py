import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from message_lstm import message_lstm


class my_model(nn.Module):
    def __init__(self,
                 num_tasks: int,
                 input_size: int,
                 hidden_size: int,
                 message_size: int = None,
                 bias: bool = True,
                 batch_first: bool = True,
                 bidirectional: bool = False):
        super(my_model, self).__init__()

        self.num_tasks = num_tasks
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.message_size = message_size
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.share_lstm = nn.LSTM(self.input_size, self.hidden_size,
                                  num_layers=1, batch_first=self.batch_first, bidirectional=self.bidirectional)
        self.task_specific_lstm_list = [message_lstm(self.input_size,
                                                     self.hidden_size,
                                                     self.message_size,
                                                     self.bias,
                                                     self.batch_first,
                                                     self.bidirectional)] * self.num_tasks

        self.aggregation_activation = nn.Tanh()
        self.Ws = nn.Linear(self.input_size + 2 * self.hidden_size, self.hidden_size)
        self.Us = nn.Linear(self.hidden_size, 1)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, x, TASK):

        padded_input = rnn.pad_sequence(x)
        embeddings = self.embed(padded_input)

        shared_task_output, (h_shared, c_n) = self.share_lstm(embeddings)
        h_task = torch.zero(embeddings.size(0), h_shared.size(1), self.hidden_size)
        state_h = torch.zeros(embeddings.size(0), self.hidden_size)
        state_c = torch.zeros(embeddings.size(0), self.hidden_size)

        message_lstm = self.task_specific_lstm_list[TASK]
        outputs = []
        # loop for the "decoder" or task specific layer
        for t in range(h_shared.size(1)):
            catted_input = torch.cat((h_shared, h_task, embeddings), dim = 2)
            var1 = self.Ws(catted_input) # var 1 is B x T x H
            var2 = self.aggregation_activation(var1) # var 1 is B x T x H
            Si = self.Us(var2) # Si is B x T x 1
            B = self.softmax(Si) # B is B x T x 1, softmax over T
            norm_h_shared = torch.mul(B, h_shared) # (B x T x 1 ) x (B x T x H) -> (B x T x H)
            Rt = torch.sum(norm_h_shared, dim = 1) # (B x H)
            output, state_h, state_c = self.task_specific_lstm_list[TASK]._step(embeddings[t], Rt, state_h, state_c)
            outputs.append(output)
            h_task = state_h.repeat(1, h_shared.size(1), 1) # state_h was B x H -> B x T x H

        out = torch.stack(outputs, axis=1)

        return out

