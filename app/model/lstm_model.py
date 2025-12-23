import torch
import torch.nn as nn

# ----------- Model Definition -----------
class LSTMTripDuration(nn.Module):
    def __init__(self, num_input_features):
        super().__init__()

        self.hour_emb = nn.Embedding(24, 4)
        self.day_emb = nn.Embedding(7, 3)
        self.month_emb = nn.Embedding(12, 4)

        emb_size = 4 + 3 + 4  # total embedding size = 11
        lstm_input_size = num_input_features + emb_size

        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=128, num_layers=3,
                            batch_first=True, dropout=0.3)

        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x_num, hour, day, month):
        h_emb = self.hour_emb(hour)
        d_emb = self.day_emb(day)
        m_emb = self.month_emb(month)

        x = torch.cat([x_num, h_emb, d_emb, m_emb], dim=1)
        x = x.unsqueeze(1)  # Add time dimension
        lstm_out, _ = self.lstm(x)
        out = lstm_out[:, -1, :]  # Take output from last time step
        return self.fc(out)