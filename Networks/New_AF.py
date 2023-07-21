
import torch.nn as nn
from Networks import AF_Tools

class new_af(nn.Module):
    def __init__(self, op, af1, af2, Activation_Functions=AF_Tools.Activation_Functions_):
        self.Activation_Functions = Activation_Functions
        self.op = op
        self.af1 = af1
        self.af2 = af2
        super().__init__()

    def forward(self, x):
        if self.op < 2:
            res = self.op * self.Activation_Functions[self.af1](x)
        elif self.op == 2:
            res1 = self.Activation_Functions[self.af2](x)
            res = self.Activation_Functions[self.af1](res1)
        elif self.op == 3:
            res = self.Activation_Functions[self.af1](x) + self.Activation_Functions[self.af2](x)
        elif self.op == 4:
            res = self.Activation_Functions[self.af1](x) * self.Activation_Functions[self.af2](x)
        positive = [x > 0]
        res[positive] = x[positive]
        return res
