import torch
import torch.nn.functional as F

class FCNN(torch.nn.Module):
    def __init__(self, n_in, l_1, l_2, n_out):
        super().__init__()

        self.fc1 = torch.nn.Linear(n_in, l_1)
        self.fc2 = torch.nn.Linear(l_1, l_2)
        self.fc3 = torch.nn.Linear(l_2, n_out)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y
