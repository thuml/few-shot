import numpy as np
import torch
from torch import nn as nn
from torch.nn import functional as F

from .baseline_metrics import MarginMetric
from few_shot.utils import get_few_shot_label


class BaselineTrain(nn.Module):

    def __init__(self, backbone, num_class, metric_type, metric_params):
        super(BaselineTrain, self).__init__()
        self.backbone = backbone
        self.metric_type = metric_type
        self.classifier = MarginMetric(metric_type, self.backbone.final_feat_dim, num_class, **metric_params)

    def forward(self, x, y=None):
        feature = self.backbone(x)
        scores = self.classifier(feature, y)
        return scores, y

    @staticmethod
    def calculate_loss(scores, y):
        return F.cross_entropy(scores, y)

class BaselineFinetune(nn.Module):
    def __init__(self, n_way, n_support, metric_type, metric_params, finetune_params):
        super(BaselineFinetune, self).__init__()
        self.n_way = n_way
        self.n_support = n_support
        self.metric_type = metric_type
        self.metric_params = metric_params
        self.finetune_params = finetune_params

    def forward(self, z_all, y):
        z_all = z_all.cuda()
        z_support = z_all[:, :self.n_support, :]
        z_query = z_all[:, self.n_support:, :]

        feature_dim = z_support.shape[-1]
        z_support = z_support.contiguous().view(-1, feature_dim)
        z_query = z_query.contiguous().view(-1, feature_dim)
        y_support = get_few_shot_label(self.n_way, self.n_support).cuda()
        linear_clf = MarginMetric(self.metric_type, feature_dim, self.n_way, **self.metric_params).cuda()

        if self.finetune_params.optim == "SGD":
            finetune_optimizer = torch.optim.SGD(linear_clf.parameters(), **self.finetune_params.sgd_params)
        else:
            raise ValueError(f"finetune optimization not supported: {self.finetune_params.optim}")

        criterion = nn.CrossEntropyLoss().cuda()

        batch_size = 4
        support_size = self.n_way * self.n_support
        for _ in range(self.finetune_params.iter):
            rand_id = np.random.permutation(support_size)
            for i in range(0, support_size, batch_size):

                selected_id = torch.from_numpy(rand_id[i: min(i+batch_size, support_size)]).cuda()
                z_batch = z_support[selected_id]
                y_batch = y_support[selected_id]
                scores = linear_clf(z_batch, y_batch)
                loss = criterion(scores, y_batch)

                finetune_optimizer.zero_grad()
                loss.backward()
                finetune_optimizer.step()

        with torch.no_grad():
            scores = linear_clf(z_query)
        return scores, y




