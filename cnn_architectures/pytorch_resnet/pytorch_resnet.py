# -*- coding: utf-8 -*-
"""
From scratch implementation of the famous ResNet models.
The intuition for ResNet is simple and clear, but to code
it didn't feel super clear at first, even when reading Pytorch own
implementation.
"""

import torch
import torch.nn as nn

class block(nn.Module):
    def __init__(self, channels, intermediate_channels, identity_dowm):