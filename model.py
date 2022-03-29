import torch
import torch.nn as nn
import math

class FFNN(nn.Module):
    def __init__(self, vocab_size):
        super(FFNN, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(vocab_size * 20, 1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, 100),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(100, 19),
        )

    def forward(self, x):
        feature_vector = x.view(x.size(0), -1)
        logits = self.classifier(feature_vector)
        return logits


class CustomFFNN(nn.Module):
    def __init__(self, vocab_size):
        super(CustomFFNN, self).__init__()
        self.layer1 = self.make_layer(vocab_size * 20, 1000)
        self.BN1 = nn.BatchNorm1d(1000)
        self.layer2 = self.make_layer(1000, 100)
        self.BN2 = nn.BatchNorm1d(100)
        self.layer3 = self.make_layer(100, 19)
        self.ReLU = nn.ReLU(True)
        self.Dropout = nn.Dropout()
        self.Sigmoid = nn.Sigmoid()
    def forward(self, x):
        feature_vector = x.view(x.size(0), -1)
        feature_vector1 = torch.mm(feature_vector, self.layer1)
        feature_vector1 = self.ReLU(feature_vector1)
        feature_vector1 = self.Dropout(feature_vector1)
        feature_vector2 = torch.mm(feature_vector1, self.layer2)
        feature_vector2 = self.ReLU(feature_vector2)
        feature_vector2 = self.Dropout(feature_vector2)
        logits = torch.mm(feature_vector2, self.layer3)
        return logits

    def make_layer(self, in_features, out_features):
        linear = nn.Parameter(torch.Tensor(in_features, out_features))
        nn.init.kaiming_uniform_(linear, a=math.sqrt(5))
        return linear