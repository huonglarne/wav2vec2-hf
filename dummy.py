import torch

from torch import nn

conv = nn.Conv1d( # this one
        1024,
        1024,
        kernel_size=128,
        padding=64,
        groups=16,
    )

conv.weight = torch.load("conv_weight.pt")

conv = nn.utils.weight_norm(conv, name="weight", dim=2)

conv = conv.cuda()

hidden_states = torch.load("hidden_states.pt")

hidden_states = conv(hidden_states)