import torch.nn as nn

class SignLSTM(nn.Module):
    def __init__(self):
        super(SignLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=63, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 3)  # 3 classes: hello, thanks, yes

    def forward(self, x):
        # x shape: (batch, 30, 63)
        # out shape: (batch, 30, 64)
        out, _ = self.lstm(x)

        # only care about the last time step (did they finish signing?)
        # last_out shape: (batch, 64)
        last_out = out[:, -1, :]

        return self.fc(last_out)  # shape: (batch, 3)