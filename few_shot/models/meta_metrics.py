from abc import abstractmethod

import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from few_shot.backbones.conv4 import init_layer

__all__ = ["ProtoNet", "MatchingNet", "RelationNet"]


class MetaBaseMetric(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, z_support, z_query):
        """
        inputs:
            z_support: shape (n_way, n_support, feature_dim)
            z_query: shape (n_way * n_query, feature_dim)
        return:
            scores: shape (n_way * n_query, n_way)
        """
        pass

    def calculate_loss(self, scores, y):
        return F.cross_entropy(scores, y)


class ProtoNet(MetaBaseMetric):
    def forward(self, z_support, z_query):
        # shape (n_way, feature_dim)
        z_proto = z_support.mean(1)
        # shape (n_way*n_query, n_way)
        dists = self.pairwise_euclidean_dist_square(z_query, z_proto)
        return - dists

    @staticmethod
    def pairwise_euclidean_dist_square(a, b):
        """
        a: shape (N, D)
        b: shape (M, D)
        output: shape (N, M)
        """
        a = a.unsqueeze(1)
        b = b.unsqueeze(0)
        return torch.pow(a - b, 2).sum(2)


class MatchingNet(MetaBaseMetric):
    def __init__(self, scale_factor=100, **kwargs):
        self.scale_factor = scale_factor
        super().__init__(**kwargs)

    def forward(self, z_support, z_query):
        n_way, n_support, feature_dim = z_support.shape

        # (n_way * n_query, n_way * n_support)
        cosine_distance = F.linear(F.normalize(z_query.reshape(-1, feature_dim)),
                                   F.normalize(z_support.reshape(-1, feature_dim)))
        prob = F.softmax(cosine_distance * self.scale_factor, dim=-1)
        return prob.reshape(-1, n_way, n_support).sum(-1)

    def calculate_loss(self, scores, y):
        return F.nll_loss(scores.log(), y)


class RelationNet(MetaBaseMetric):
    def __init__(self, feature_dims, **kwargs):
        super().__init__(**kwargs)
        self.feature_dims = feature_dims
        # relation net features are not pooled, so self.feat_dim is [c, h, w]
        self.relation_module = RelationModule(self.feature_dims, 8)

    def forward(self, z_support, z_query):
        """
        inputs:
            z_support: shape (n_way, n_support, c*h*w)
            z_query: shape (n_way * n_query, c*h*w)
        return:
            scores: shape (n_way * n_query, n_way)
        """

        n_way, n_support = z_support.shape[:2]
        query_size = z_query.shape[0]
        c, h, w = self.feature_dims
        assert z_support.shape[-1] == c*h*w
        assert z_query.shape[-1] == c*h*w

        # shape (n_way, c, h, W)
        z_proto = z_support.contiguous().view(n_way, n_support, c, h, w).mean(1)
        # shape (n_way * n_query, c, h, W)
        z_query = z_query.contiguous().view(query_size, c, h, w)

        # shape (n_way*n_query, n_way, c, h, W)
        z_proto_ext = z_proto.unsqueeze(0).repeat(query_size, 1, 1, 1, 1)
        z_query_ext = z_query.unsqueeze(1).repeat(1, n_way, 1, 1, 1)
        # shape (n_way*n_query, n_way, c, h, W)
        relation_pairs = torch.cat((z_proto_ext, z_query_ext), 2).view(query_size * n_way, c * 2, h, w)
        scores = self.relation_module(relation_pairs).view(query_size, n_way)
        return scores


class RelationConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, padding=0):
        super(RelationConvBlock, self).__init__()
        self.indim = input_dim
        self.outdim = output_dim
        self.C = nn.Conv2d(input_dim, output_dim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(output_dim, momentum=1, affine=True)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)

        self.parametrized_layers = [self.C, self.BN, self.relu, self.pool]

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out


class RelationModule(nn.Module):
    """docstring for RelationNetwork"""

    def __init__(self, input_size, hidden_size):
        super(RelationModule, self).__init__()

        # when using Resnet, conv map without avgpooling is 7x7, need padding in block to do pooling
        padding = 1 if (input_size[1] < 10) and (input_size[2] < 10) else 0

        self.layer1 = RelationConvBlock(input_size[0] * 2, input_size[0], padding=padding)
        self.layer2 = RelationConvBlock(input_size[0], input_size[0], padding=padding)

        shrink_s = lambda s: int((int((s - 2 + 2 * padding) / 2) - 2 + 2 * padding) / 2)

        self.fc1 = nn.Linear(input_size[0] * shrink_s(input_size[1]) * shrink_s(input_size[2]), hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = self.fc2(out)

        return out
