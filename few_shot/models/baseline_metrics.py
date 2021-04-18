import math

import torch
from torch import nn as nn
from torch.nn import Parameter, functional as F

from few_shot.utils import one_hot


class MarginMetric(nn.Module):
    r"""Implement of large margin softmax/cosine distance: :
    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        scale_factor: norm of input feature
        margin: margin, if set to 0, no margin is added
        metric_type: softmax / cosine / net-softmax / net-cosine
    :return： logit
    """

    def __init__(self, metric_type, in_features, out_features, scale_factor=30.0, margin=0.0):
        if "neg" in metric_type:
            assert margin <= 0, f"margin = {margin} should <= 0"
        if "neg" not in metric_type:
            assert margin == 0, f"margin = {margin} should be 0"

        super(MarginMetric, self).__init__()
        self.metric_type = metric_type
        self.scale_factor = scale_factor
        self.margin = margin
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, feature, label=None):
        if "cosine" in self.metric_type:
            sim = F.linear(F.normalize(feature), F.normalize(self.weight))
        elif "softmax" in self.metric_type:
            sim = F.linear(feature, self.weight)
            sim -= sim.min(dim=1, keepdim=True)[0]
        else:
            raise ValueError(f"metric type {self.metric_type} not supported")

        # for training and negative margin metric, add margin to the sim
        if label is not None and "neg" in self.metric_type:
            phi = sim - self.margin
            sim = torch.where(one_hot(label, sim.shape[1]).byte(), phi, sim)

        return sim * self.scale_factor
