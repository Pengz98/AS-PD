import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.regis_utils.qdataset import QuaternionTransform, rad_to_deg


class TaskLoss_cls(nn.Module):
    def __init__(self):
        super(TaskLoss_cls, self).__init__()

    def forward(self, pred, target):
        loss = F.nll_loss(pred, target)
        return loss

class TaskLoss_regis(nn.Module):
    def __init__(self):
        super(TaskLoss_regis, self).__init__()

    def forward(self, samp_data, task_model):
        pcrnet_loss, pcrnet_loss_info = compute_pcrnet_loss(
            task_model, samp_data
        )
        return pcrnet_loss, pcrnet_loss_info

class TaskLoss_recon(nn.Module):
    def __init__(self):
        super(TaskLoss_recon, self).__init__()

    def forward(self, pred, gt):
        loss = ReconstrLoss()(pred, gt)
        return loss

class ReconstrLoss(nn.Module):
    def __init__(self, mean=True):
        super(ReconstrLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.mean = mean

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        preds, gts = preds.double(), gts.double()
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)

        if self.mean:
            loss_1 = torch.mean(mins,1)
        else:
            loss_1 = torch.sum(mins,1)
        loss_1 = torch.mean(loss_1)

        mins, _ = torch.min(P, 2)
        if self.mean:
            loss_2 = torch.mean(mins,1)
        else:
            loss_2 = torch.sum(mins,1)
        loss_2 = torch.mean(loss_2)

        return loss_1 + loss_2


class SimplifyLoss(nn.Module):
    def __init__(self, beta, gamma, delta, mean=True):
        super(SimplifyLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.mean = mean
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def batch_pairwise_dist(self, x, y):
        bs, num_points_x, points_dim = x.size()
        _, num_points_y, _ = y.size()
        xx = x.pow(2).sum(dim=-1)
        yy = y.pow(2).sum(dim=-1)
        zz = torch.bmm(x, y.transpose(2, 1))
        rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
        ry = yy.unsqueeze(1).expand_as(zz)
        P = (rx.transpose(2, 1) + ry - 2 * zz)
        return P

    def forward(self, preds, gts):
        preds, gts = preds.double(), gts.double()
        P = self.batch_pairwise_dist(gts, preds)
        num_generated = P.shape[2]

        mins, _ = torch.min(P, 1)

        loss_m, _ = torch.max(mins,1)
        loss_m = torch.mean(loss_m)

        if self.mean:
            loss_1 = torch.mean(mins,1)
        else:
            loss_1 = torch.sum(mins,1)
        loss_1 = torch.mean(loss_1)

        mins, _ = torch.min(P, 2)
        if self.mean:
            loss_2 = torch.mean(mins,1)
        else:
            loss_2 = torch.sum(mins,1)
        loss_2 = torch.mean(loss_2)

        return loss_1 + (self.gamma+self.delta*num_generated)*loss_2 + self.beta*loss_m

def compute_pcrnet_loss(model, data, device='cuda' if torch.cuda.is_available() else 'cpu', sup=True):
    p0, p1, igt = data
    p0 = p0.to(device)  # template
    p1 = p1.to(device)  # source
    # igt = igt.to(device) # igt: p0 -> p1

    twist, pre_normalized_quat = model(p0, p1)

    # https://arxiv.org/pdf/1805.06485.pdf QuaterNet quaternient regularization loss
    qnorm_loss = torch.mean((torch.sum(pre_normalized_quat ** 2, dim=1) - 1) ** 2)

    est_transform = QuaternionTransform(twist)
    gt_transform = QuaternionTransform.from_dict(igt, device)

    p1_est = est_transform.rotate(p0)

    chamfer_loss = ReconstrLoss()(p1, p1_est)

    rot_err, norm_err, trans_err = est_transform.compute_errors(gt_transform)

    if sup:
        pcrnet_loss = 1.0 * norm_err + 1.0 * chamfer_loss

    else:
        pcrnet_loss = chamfer_loss

    rot_err = rad_to_deg(rot_err)

    pcrnet_loss_info = {
        "chamfer_loss": chamfer_loss,
        "qnorm_loss": qnorm_loss,
        "rot_err": rot_err,
        "norm_err": norm_err,
        "trans_err": trans_err,
        "est_transform": est_transform,
        "transform_source": p1_est
    }

    return pcrnet_loss, pcrnet_loss_info