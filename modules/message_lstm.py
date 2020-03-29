import torch
from torch import nn


class message_lstm(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, message_size: int = None, bias: bool = True, batch_first: bool = True, bidirectional: bool = False):
        """
        这就是大佬lstm， 只有大佬能够用

        In the constructor we instantiate two nn.Linear modules and assign them as
        member variables.

        input_size – The number of expected features in the input x

        hidden_size – The number of features in the hidden state h

        message_size – The number of features in the aggregated message r

        num_layers – Number of recurrent layers. E.g., setting num_layers=2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results. Default: 1

        bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True

        batch_first – If True, then the input and output tensors are provided as (batch, seq, feature). Default: False

        dropout – If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout. Default: 0

        bidirectional – If True, becomes a bidirectional LSTM. Default: Fal
        """
        super(message_lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        if message_size is not None:
            self.message_size = message_size
        else:
            self.message_size = hidden_size
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.Ws = nn.Linear(self.input_size + 2 * self.hidden_size, self.hidden_size)
        self.Us = nn.Linear(self.hidden_size, 1)
        self.aggregate_activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        # input gate
        self.Wii = nn.Linear(input_size, hidden_size, bias = bias)
        self.Whi = nn.Linear(hidden_size, hidden_size, bias = bias)
        self.input_activation = nn.Sigmoid()

        # forget gate
        self.Wif = nn.Linear(input_size, hidden_size, bias=bias)
        self.Whf = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.forget_activation = nn.Sigmoid()

        # cell gate
        self.Wig = nn.Linear(input_size, hidden_size, bias = bias)
        self.Whg = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.cell_activation = nn.Tanh()

        # output gate
        self.Wio = nn.Linear(input_size, hidden_size, bias = bias)
        self.Who = nn.Linear(hidden_size, hidden_size, bias = bias)
        self.output_activation = nn.Sigmoid()

        # message gate
        self.Wrm = nn.Linear(self.message_size, hidden_size)
        self.Wcm = nn.Linear(hidden_size, hidden_size)
        self.message_activation = nn.Sigmoid()
        self.Wr = nn.Linear(self.message_size, hidden_size, bias=False)  # no bias here according to Liu et al.

        self.hidden_activation = nn.Tanh()

    def _step(self, x_slice, r = None, state_h = None, state_c = None):
        """
        :params:
            x_slice: a B x input_size dimensional matrix (slice of a B x T x input_size tensor)
            r: aggregated message passed in via the neural message passing framework

        :return:
            output: hidden_state at timestep t
            state_h: hidden_state after 1 iteration
            state_c: cell state after 1 iterateion
        """

        if state_h is None and state_c is None:
            batch_size = x_slice.shape[0]
            state_c = torch.zeros((batch_size, self.hidden_size))
            state_h = torch.zeros((batch_size, self.hidden_size))

        i = self.input_activation(self.Wii(x_slice) + self.Whi(state_h))
        f = self.forget_activation(self.Wif(x_slice) + self.Whf(state_h))
        g = self.cell_activation(self.Wig(x_slice) + self.Whg(state_h))
        o = self.output_activation(self.Wio(x_slice) + self.Who(state_h))

        state_c = torch.mul(f, state_c) + torch.mul(i, g)
        if r is None:
            h_tilde = self.hidden_activation(state_c)
            state_h = torch.mul(o, h_tilde)
        else:
            m = self.message_actvation(self.Wrm(r) + self.Wcm(state_c))
            R = self.Wr(r)
            M = torch.mul(m, R)
            h_tilde = self.hidden_activation(state_c + M)
            state_h = torch.mul(o, h_tilde)
        return state_h, state_h, state_c


    def forward(self, x):
        """

        """
        if self.batch_first is False:
            x = x.transpose(0,1)
        state_h = state_c = None
        outputs = []
        for i in range(x.shape[1]):
            output, state_h, state_c = self._step(x[:, i, :], state_h = state_h, state_c = state_c)
            outputs.append(output)

        out = torch.stack(outputs, axis = 1)

        return out


