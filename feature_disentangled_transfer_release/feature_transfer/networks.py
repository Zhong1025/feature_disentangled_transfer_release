"""Shared and dataset-specific encoders for cross-system representation learning."""
import torch
from torch import nn

class DatasetSpecificEncoder(nn.Module):
    def __init__(self, d_in, hidden_dims):
        super(DatasetSpecificEncoder, self).__init__()
        layers = []
        prev_dim = d_in
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, d_in))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SharedEncoder(nn.Module):
    def __init__(self, d_in, hidden_dims):
        super(SharedEncoder, self).__init__()
        layers = []
        prev_dim = d_in
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, d_in))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SharedSpecificRepresentation(nn.Module):
    def __init__(self, d_in=128, hidden_dims=None):
        super().__init__()
        # 领域相关特征提取器
        if hidden_dims is None:
            hidden_dims = [256, 128]
        self.domain_specific_T = DatasetSpecificEncoder(d_in, hidden_dims)
        self.domain_specific_S = DatasetSpecificEncoder(d_in, hidden_dims)
        # 领域共享特征提取器
        self.domain_shared = SharedEncoder(d_in, hidden_dims)

    def forward(self, e_T, e_S):
        assert e_T.shape[1] == e_S.shape[1], "Source/Target feature dimensions must match"
        h_specific_T = self.domain_specific_T(e_T)
        h_specific_S = self.domain_specific_S(e_S)
        h_shared_T = self.domain_shared(e_T)
        h_shared_S = self.domain_shared(e_S)
        return h_specific_T, h_specific_S, h_shared_T, h_shared_S


class ReconstructionNetwork(nn.Module):
    def __init__(self, d_in=16, h_specific_dim=16, h_share_dim=16):
        super().__init__()
        # 增加一层隐藏层，提高重构能力
        self.fc1 = nn.Linear(h_specific_dim + h_share_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, d_in)

    def forward(self, h_specific, h_share):
        combined = torch.cat((h_specific, h_share), dim=1)
        x = self.relu1(self.bn1(self.fc1(combined)))
        x = self.relu2(self.bn2(self.fc2(x)))
        return self.fc3(x)


class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class DomainDiscriminator(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h_share):
        h_share = GradientReversalLayer.apply(h_share, 1.0)
        return self.sigmoid(self.fc(h_share))

