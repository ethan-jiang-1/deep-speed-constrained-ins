import torch

# Model

class VelRegressorLstm(torch.nn.Module):
    def __init__(self, Nin=200, Nout=1, batch_size=10, device=None,
                 lstm_size=100, lstm_layers=6, dropout=0):
        """
        Simple LSTM network
        Input: torch array [batch x frames x input_size]
        Output: torch array [batch x frames x out_size]

        :param input_size: num. channels in input
        :param out_size: num. channels in output
        :param batch_size:
        :param device: torch device
        :param lstm_size: number of LSTM units per layer
        :param lstm_layers: number of LSTM layers
        :param dropout: dropout probability of LSTM (@ref https://pytorch.org/docs/stable/nn.html#lstm)
        """
        super(VelRegressorLstm, self).__init__()
        self.input_size = Nin
        self.lstm_size = lstm_size
        self.output_size = Nout
        self.num_layers = lstm_layers
        self.batch_size = batch_size
        self.device = device

        self.lstm = torch.nn.LSTM(
            self.input_size, self.lstm_size, self.num_layers, batch_first=True, dropout=dropout)
        self.linear1 = torch.nn.Linear(self.lstm_size, self.output_size * 60)
        self.linear2 = torch.nn.Linear(6 * 60, 60)
        self.linear3 = torch.nn.Linear(60, 3)
        self.linear4 = torch.nn.Linear(3, self.output_size)
        #self.hidden = self.init_weights()

    def forward(self, input, hidden=None):
        # input tensor shape (10, 6, 200) in batch_mode(10)
        output, _ = self.lstm(input)
        output = self.linear1(output)
        output = output.view(output.size(0), -1)
        output = self.linear2(output)
        output = self.linear3(output)
        output = self.linear4(output)
        return output
