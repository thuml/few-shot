import torch.nn as nn
from . import meta_metrics
from ..utils import get_few_shot_label


class Meta(nn.Module):
    def __init__(self, backbone, n_support, metric_type, metric_params):
        super(Meta, self).__init__()
        self.n_support = n_support
        self.backbone = backbone
        self.feature_dims = self.backbone.final_feat_dim
        self.linear_clf = meta_metrics.__dict__[metric_type](feature_dims=self.feature_dims, **metric_params)

    def forward(self, x, y=None):
        """
        inputs:
            x: shape (n_way, n_support+n_query, **image_size or feature_dim)
            y: shape (n_way, n_support+n_query), not used, for compatibility
        return:
            scores: (n_way * n_query, num_class)
            y_query: (n_way * n_query, )
        """
        is_feature = x.shape[2] > 3
        z_support, z_query = self.parse_feature(x, is_feature=is_feature)
        n_way, n_query, feature_dim = z_query.shape

        scores = self.linear_clf(z_support, z_query.reshape(n_way*n_query, feature_dim))
        y_query = get_few_shot_label(n_way, n_query).cuda(non_blocking=True)
        return scores, y_query

    def calculate_loss(self, scores, y):
        return self.linear_clf.calculate_loss(scores, y)

    def parse_feature(self, x, is_feature):
        n_way, n_all = x.shape[:2]
        if is_feature:
            feature = x
        else:
            img_size = x.shape[2:]
            # shape (n_way, n_support + n_query, *img_size)
            x = x.contiguous().view(-1, *img_size)
            # shape (n_way, n_support + n_query, feature_dim)
            feature = self.backbone(x)
        z_all = feature.view(n_way, n_all, -1)
        z_support = z_all[:, :self.n_support, :]
        z_query = z_all[:, self.n_support:, :]
        return z_support, z_query
