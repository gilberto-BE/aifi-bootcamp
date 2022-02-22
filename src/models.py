import torch.nn as nn
import torch.nn.functional as F

class NeuralNetwork(nn.Module):
    def __init__(self, n_features, n_hidden, n_outputs):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(n_features, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_hidden)
        self.pred = nn.Linear(n_hidden, n_outputs)


    def forward(self, x):
        x = self.flatten(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = self.pred(x)