import torch
import torch.nn as nn

class ConvNet(nn.Module):

    def __init__(self):
        super(ConvNet, self).__init__()

        model_list = [
            nn.Conv2d(1, 4, 5),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(4, 12, 5),
            nn.ReLU(),
            nn.MaxPool2d(2)
        ]

        self.model = nn.Sequential(*model_list)
        self.fc = nn.Linear(12*4*4, 10)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        out = self.model(x)
        out = out.view(x.size(0), -1)
        out = self.fc(out)
        return out
