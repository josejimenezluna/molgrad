import torch
import torch.nn as nn
import torch.functional as F


if __name__ == "__main__":
    x = torch.randn((1, 25), requires_grad=True)

    linear1 = nn.Linear(25, 25)
    linear2 = nn.Linear(25, 4)
    y = torch.relu(linear1(x))
    z = torch.relu(linear2(x))
    w = z[0, 2]
    w.backward(retain_graph=True)
    u = z[0, 3]
    u.backward()
