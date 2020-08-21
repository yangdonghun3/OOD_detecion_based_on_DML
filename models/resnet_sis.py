import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self, embedding_net):
        super(ResNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        output = self.embedding_net(x)

        return output

    def get_embedding(self, x):
        return self.embedding_net(x)