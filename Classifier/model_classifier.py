import torch
from torch import nn


class TrajectoryClassifier(nn.Module):
    def __init__(self, device, input_dim=2, hidden_dim=128, num_layers=2):
        """
        Classifies whether a given trajectory sequence belongs to a controlled (real) or random (fake) movement.

        input_dim: 2 (xy coordinates)
        hidden_dim: LSTM hidden size
        num_layers: number of LSTM layers
        """
        super(TrajectoryClassifier, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            # nn.Sigmoid()
        )

    def forward(self, x):
        """
        x: (Batch, N, Seq, 2) - Trajectories for all agents
        """
        batch_size, num_agents, seq_length, _ = x.shape
        x = x.view(batch_size * num_agents, seq_length, -1)
        # with torch.backends.cudnn.flags(enabled=False):
        _, (hidden, _) = self.lstm(x)  #
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  #
        hidden = hidden.view(batch_size, num_agents, -1)
        x = hidden
        x = self.fc(x)
        x = x.squeeze(-1)
        return x