import torch
from torch import nn
from torch.nn import functional as F


class HausdorffLoss(nn.Module):
    def __init__(self, dist_func):
        super(HausdorffLoss, self).__init__()
        if dist_func == 'L1':
            self.loss_func = self.l1_dist
        elif dist_func == 'L2':
            self.loss_func = self.l2_dist
        elif dist_func == 'Cosine':
            self.loss_func = self.cosine_dist
        else:
            self.loss_func = dist_func

    @staticmethod
    def l2_dist(x, y):
        differences = x.unsqueeze(1) - y.unsqueeze(0)
        distances = torch.mean(differences ** 2, -1)
        return distances

    @staticmethod
    def l1_dist(x, y):
        differences = x.unsqueeze(1) - y.unsqueeze(0)
        distances = torch.mean(differences.abs(), -1)
        return distances

    @staticmethod
    def cosine_dist(x, y):
        sim_matrix = F.cosine_similarity(x.unsqueeze(1), y.unsqueeze(0), dim=-1)
        return 1 - sim_matrix

    def forward(self, set1, set2):
        dist_matrix = self.loss_func(set1, set2)
        t1 = torch.mean(torch.min(dist_matrix, 1)[0])
        t2 = torch.mean(torch.min(dist_matrix, 0)[0])
        return (t1 + t2) / 2


