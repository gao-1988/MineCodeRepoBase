import torch
import numpy as np
from torch import nn


def chamfer_loss(a, b):
    """ return Chamfer distance
    Args:
        a:
        b:
    Returns:
    """
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    # diag_ind = torch.arange(0, num_points).type(torch.cuda.LongTensor)
    diag_ind = torch.arange(0, num_points)
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return torch.mean(torch.min(P, 1)[0]) + torch.mean(torch.min(P, 2)[0])


class ChamferLoss(nn.Module):

    def __init__(self):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, preds, gts):
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        # CHANGED torch.sum -> torch.mean
        loss_1 = torch.mean(mins)
        mins, _ = torch.min(P, 2)
        loss_2 = torch.mean(mins)
        return loss_1 + loss_2

    def batch_pairwise_dist(self, x, y):
        #  [batch, n, 3]
        x = x.float()
        y = y.float()
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = torch.bmm(x, x.transpose(2, 1))
        yy = torch.bmm(y, y.transpose(2, 1))
        zz = torch.bmm(x, y.transpose(2, 1))
        if self.use_cuda:
            dtype = torch.cuda.LongTensor
        else:
            dtype = torch.LongTensor
        diag_ind_x = torch.arange(0, num_points_x).type(dtype)
        diag_ind_y = torch.arange(0, num_points_y).type(dtype)
        # brk()
        rx = xx[:, diag_ind_x, diag_ind_x].unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy[:, diag_ind_y, diag_ind_y].unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P


class CorrespondenceLoss(torch.nn.Module):
    def forward(self, template, source, corr_mat_pred, corr_mat):
        # corr_mat:			batch_size x num_template x num_source (ground truth correspondence matrix)
        # corr_mat_pred:	batch_size x num_source x num_template (predicted correspondence matrix)
        batch_size, _, num_points_template = template.shape
        _, _, num_points = source.shape
        return torch.nn.functional.cross_entropy(corr_mat_pred.view(batch_size * num_points, num_points_template),
                                                 torch.argmax(corr_mat.transpose(1, 2).reshape(-1, num_points_template),
                                                              axis=1))

