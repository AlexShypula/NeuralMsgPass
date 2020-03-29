import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
from .message_lstm import message_lstm


class direct_communication_network(nn.Module):
    def __init__(self,
                 vocab,
                 embed_dim: int,
                 num_tasks: int,
                 input_size: int,
                 hidden_size: int,
                 message_size: int = None,
                 bias: bool = True,
                 batch_first: bool = True,
                 bidirectional: bool = False):
        super(my_model, self).__init__()

        self.num_tasks = num_tasks
        self.vocab_size = len(vocab)
        self.embed_dim = embed_dim
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.message_size = message_size
        self.bias = bias
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.embed = nn.Embedding(self.input_size, self.embed_dim)
        # self.embed = nn.Embedding.from_pretrained(vocab)
        # self.embed.weight.data.copy_(vocab.vectors)

        self.task_specific_lstm_list = [message_lstm(self.embed_dim,
                                                     self.hidden_size,
                                                     self.message_size,
                                                     self.bias,
                                                     self.batch_first,
                                                     self.bidirectional)] * self.num_tasks

        self.aggregation_activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x, TASK):

        padded_input = rnn.pad_sequence(x)
        embeddings = self.embed(padded_input)

        shared_task_output, (h_shared, c_n) = self.share_lstm(embeddings)

        h_task = torch.zero(embeddings.size(0), embeddings.size(1), self.hidden_size)
        state_h = torch.zeros(embeddings.size(0), self.hidden_size)
        state_c = torch.zeros(embeddings.size(0), self.hidden_size)

        aggregated_messages = [] # will collect each of the messages in Time x B x H (so need to transpose the outputs of the lstms)
        for task_index in range(len(self.task_specific_lstm_list)):
            if task_index != TASK:
                output = self.task_specific_lstm_list[task_index](embeddings)
                aggregated_messages.append(output.transpose(0,1)) # make it Time x B x H

        aggregated_messages = torch.stack(aggregated_messages, dim = 3) # aggregated messages will now be a Time x B x H x Task (for BMM)

        task_specific_lstm = self.task_specific_lstm_list[TASK]
        outputs = []
        # loop for the "decoder" or task specific layer
        for t in range(h_shared.size(1)):
            aggregated_t = aggregated_messages[t] # B x H x Task
            # optimized matmul here
            var1_p1 = torch.matmul(task_specific_lstm.Ws_p1, aggregated_t) # Ws is H x H, aggregated_t is B x H x Task so var 1 is B x H x Task
            var1_p2 = torch.matmul(torch.cat((state_h, embeddings[:,t,:]), dim = 1), task_specific_lstm.Ws_p2) # (B x [H+E]) x ([H+E] x H) -> B x H
            var1 = var1_p1 + var1_p2.unsqueeze(2) # add extra dimension to p2 variable for broadcasting
            var2 = task_specific_lstm.aggregation_activation(var1)  # var 2 is B x H x Task
            # transpose var2 so that it is B x Task x H (we the want to collapse down the H with the linear transform)
            Si = task_specific_lstm.Us(var2.transpose(1,2))  # Si is B x Task x 1
            a = task_specific_lstm.softmax(Si)  # B is B x Task x 1, softmax over Tasks
            # transpose aggregated_t so that dimensions (B x H x Task) -> (B x Task x H)
            norm_h_shared = torch.mul(a, aggregated_t.transpose(1,2))  # (B x Task x 1 ) * (B x Task x H) -> (B x Task x H)
            Rt = torch.sum(norm_h_shared, dim=1)  # (B x H)
            output, state_h, state_c = task_specific_lstm._step(embeddings[:,t,:], Rt, state_h, state_c)
            outputs.append(output)

        out = torch.stack(outputs, axis=1)

        return out, state_h