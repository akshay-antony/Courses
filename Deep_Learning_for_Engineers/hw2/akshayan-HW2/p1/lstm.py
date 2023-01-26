from distutils.command.check import HAS_DOCUTILS
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

class FlowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=[128, 128, 128], num_layers=3, dropout=0.1):
        super(FlowLSTM, self).__init__()
        # build your model here
        # your input should be of dim (batch_size, seq_len, input_size)
        # your output should be of dim (batch_size, seq_len, input_size) as well
        # since you are predicting velocity of next step given previous one
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.time_step = 19
        self.net = []
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.lstm1 = nn.LSTMCell(input_size, hidden_size[0]).to(self.device)
        for i in range(num_layers-1):
            self.net.append(nn.LSTMCell(hidden_size[i], hidden_size[i+1]).to(self.device))
        
        self.linear = nn.Linear(hidden_size[-1], input_size).to(self.device)
        
        # feel free to add functions in the class if needed


    # forward pass through LSTM layer
    def forward(self, x):
        '''
        input: x of dim (batch_size, 19, 17)
        '''
        # define your feedforward pass
       
        output_lstm_prev = []
        output_lstm_curr = []
        output = torch.zeros((x.shape[0], 0, self.input_size)).to(self.device)
        for i in range(x.shape[1]):
            output_lstm_curr = []
            if i == 0:
                c_0 = torch.zeros((x.shape[0], self.hidden_size[0])).to(self.device)
                h_0 = torch.zeros((x.shape[0], self.hidden_size[0])).to(self.device)
                h_1, c_1 = self.lstm1(x[:, i, :].squeeze(), (h_0, c_0))
                output_lstm_curr.append((h_1, c_1))
            else:
                h_1, c_1 = self.lstm1(x[:, i, :].squeeze(), output_lstm_prev[0])
                output_lstm_curr.append((h_1, c_1))

            for j, lstm in enumerate(self.net):
                if i == 0:
                    c0 = torch.zeros((x.shape[0], self.hidden_size[j+1])).to(self.device)
                    h0 = torch.zeros((x.shape[0], self.hidden_size[j+1])).to(self.device)
                    h_out, c_out = lstm(output_lstm_curr[-1][0], (h0, c0)) 
                    output_lstm_curr.append((h_out, c_out))
                else:
                    h_out, c_out = lstm(output_lstm_curr[-1][0].to(self.device), output_lstm_prev[j+1])
                    output_lstm_curr.append((h_out, c_out))
            output_lstm_prev = output_lstm_curr
            out = self.linear(output_lstm_curr[-1][0])
            out = torch.unsqueeze(out, 1)
            output = torch.cat([output, out], dim=1)
        
        return output

    # forward pass through LSTM layer for testing
    def test(self, x):
        '''
        input: x of dim (batch_size, 17)
        '''
        # define your feedforward pass
        output_lstm_prev = []
        output_lstm_curr = []
        output = torch.zeros((x.shape[0], 0, self.input_size)).to(self.device)
        for i in range(19):
            output_lstm_curr = []
            if i == 0:
                c_0 = torch.zeros((x.shape[0], self.hidden_size[0])).to(self.device)
                h_0 = torch.zeros((x.shape[0], self.hidden_size[0])).to(self.device)
                h_1, c_1 = self.lstm1(x, (h_0, c_0))
                output_lstm_curr.append((h_1, c_1))
            else:
                h_1, c_1 = self.lstm1(out.squeeze(), output_lstm_prev[0])
                output_lstm_curr.append((h_1, c_1))

            for j, lstm in enumerate(self.net):
                if i == 0:
                    c0 = torch.zeros((x.shape[0], self.hidden_size[j+1])).to(self.device)
                    h0 = torch.zeros((x.shape[0], self.hidden_size[j+1])).to(self.device)
                    h_out, c_out = lstm(output_lstm_curr[-1][0], (h0, c0)) 
                    output_lstm_curr.append((h_out, c_out))
                else:
                    h_out, c_out = lstm(output_lstm_curr[-1][0].to(self.device), output_lstm_prev[j+1])
                    output_lstm_curr.append((h_out, c_out))
            output_lstm_prev = output_lstm_curr
            out = self.linear(output_lstm_curr[-1][0])
            out = torch.unsqueeze(out, 1)
            output = torch.cat([output, out], dim=1)
        
        return output
