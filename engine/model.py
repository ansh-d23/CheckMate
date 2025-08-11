# model.py

import torch.nn as nn
import torch

class DocumentClassifier(nn.Module):
    def __init__(self):
        super(DocumentClassifier, self).__init__()
        self.fc1 = nn.Linear(1024, 512) 
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3) ## 3 classes

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
