import torch
from torch import nn


class LocalChannelAttention(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size % 2 == 1), "Kernel size must be odd"

        self.conv = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size - 1) // 2)
        self.GAP = nn.AvgPool2d(feature_map_size)

    def forward(self, x):
        N, C, H, W = x.shape
        att = torch.mean(x, (2,3)).reshape(N, 1, C)
        att = self.conv(att).sigmoid()
        att = att.repeat(1,C,1)
        att = torch.mean(att, 0)
        return att


class GlobalChannelAttention(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size % 2 == 1), "Kernel size must be odd"

        self.conv_q = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size - 1) // 2)
        self.conv_k = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size - 1) // 2)
        self.GAP = nn.AvgPool2d(feature_map_size)

    def forward(self, x):
        N, C, H, W = x.shape

        query = key = torch.mean(x, (2,3)).reshape(N, 1, C)
        query = self.conv_q(query).sigmoid()
        key = self.conv_q(key).sigmoid().permute(0, 2, 1)
        query_key = torch.bmm(key, query).reshape(N, -1)
        query_key = query_key.softmax(-1).reshape(-1, C, C)

        #value = torch.mean(x, (2,3)).reshape(N, C, C)
        #att = torch.bmm(value, query_key)
        #att = att.reshape(N, C, H, W)
        return torch.mean(query_key, 0)

