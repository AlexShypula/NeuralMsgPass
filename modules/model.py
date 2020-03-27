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

    def forward(self, x):
        padded_input = rnn.pack_sequence(x)

        for task_specific_lstm in self.task_specific_lstm_list:
            shared_task_output, (h_shared, c_n) = self.share_lstm(padded_input)
            St = torch.zeros(1)
            for t in range(h_shared.size(1)):
                catted_input = torch.cat((padded_input[:, t, :], shared_task_output[:, t, :], shared_task_output), dim=2)
                for sub_t in range(h_shared.size(1)):
                    var1 = torch.cat((padded_input[:, sub_t, :], shared_task_output[:, t, :], shared_task_output[:, sub_t, :]), dim=2)
                    Si= self.Us(self.aggregation_activation(self.Ws(var1)))
                    Si = torch.cat()
                beta = nn.Softmax(St)
                Rt = sum(beta * shared_task_output)


                inter_var_1 = self.Ws(catted_input)
                inter_var_2 = self.aggregation_activation(inter_var_1)
                Si = self.Us(inter_var_2)
                if St == 0:
                    St = Si
                else:
                    St = torch.cat((St, Si))
            Rt = St * h_shared
            h_task = task_specific_lstm(catted_input, Rt)

