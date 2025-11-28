import torch
import torch.nn as nn
import torch.nn.functional as F

class Connect4Net(nn.Module):
    # input: (3, 6, 7) board Tensor
    # output: 7 Q-values/logits (one per column)
    # the one with best score = best move for rl

    def __init__(self):
        super().__init__()

        # input channels:
        # current player's pieces (0), opponent's pieces (1), the player's turn--all 1s or 0s (2)
        self.convmax_stack = nn.Sequential(
            # 3 inputs
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2), #output: (32,3,3)

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2) # output: (64,1,1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128,7) # 7 columns
        )
    
    def forward(self, x):
        x = self.convmax_stack(x)
        logits = self.classifier(x)
        return logits
