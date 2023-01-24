import math
import numpy as np
import torch
from scipy.spatial.transform import Rotation
import torch.nn as nn

from regis_utils.qdataset import QuaternionTransform, rad_to_deg
from regis_utils.visu_utils import spam

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

    chamfer_loss = ChamferLoss()(p1, p1_est)

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

class ChamferLoss(nn.Module):
    def __init__(self, type='mean'):
        super(ChamferLoss, self).__init__()
        self.use_cuda = torch.cuda.is_available()
        self.type=type

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
        P = self.batch_pairwise_dist(gts, preds)
        mins, _ = torch.min(P, 1)
        if self.type=='mean':
            loss_1 = torch.mean(mins)
        else:
            loss_1 = torch.sum(mins)

        mins, _ = torch.min(P, 2)
        if self.type=='mean':
            loss_2 = torch.mean(mins)
        else:
            loss_2 = torch.sum(mins)
        return loss_1 + loss_2

def summary_metrics(r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic):
    r_mse = np.concatenate(r_mse, axis=0)
    r_mae = np.concatenate(r_mae, axis=0)
    t_mse = np.concatenate(t_mse, axis=0)
    t_mae = np.concatenate(t_mae, axis=0)
    r_isotropic = np.concatenate(r_isotropic, axis=0)
    t_isotropic = np.concatenate(t_isotropic, axis=0)

    r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic = \
        np.sqrt(np.mean(r_mse)), np.mean(r_mae), np.sqrt(np.mean(t_mse)), \
        np.mean(t_mae), np.mean(r_isotropic), np.mean(t_isotropic)
    return r_mse, r_mae, t_mse, t_mae, r_isotropic, t_isotropic

def compute_metrics(R, t, gtR, gtt):
    inv_R, inv_t = inv_R_t(gtR, gtt)
    cur_r_mse, cur_r_mae = anisotropic_R_error(R, inv_R)
    cur_t_mse, cur_t_mae = anisotropic_t_error(t, inv_t)
    cur_r_isotropic = isotropic_R_error(R, inv_R)
    cur_t_isotropic = isotropic_t_error(t, inv_t, inv_R)
    return cur_r_mse, cur_r_mae, cur_t_mse, cur_t_mae, cur_r_isotropic, \
           cur_t_isotropic

def inv_R_t(R, t):
    inv_R = R.permute(0, 2, 1).contiguous()
    inv_t = - inv_R @ t[..., None]
    return inv_R, torch.squeeze(inv_t, -1)

def anisotropic_R_error(r1, r2, seq='xyz', degrees=True):
    '''
    Calculate mse, mae euler agnle error.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    if isinstance(r1, torch.Tensor):
        r1 = r1.cpu().detach().numpy()
    if isinstance(r2, torch.Tensor):
        r2 = r2.cpu().detach().numpy()
    assert r1.shape == r2.shape
    eulers1, eulers2 = [], []
    for i in range(r1.shape[0]):
        euler1 = Rotation.from_matrix(r1[i]).as_euler(seq=seq, degrees=degrees)
        euler2 = Rotation.from_matrix(r2[i]).as_euler(seq=seq, degrees=degrees)
        eulers1.append(euler1)
        eulers2.append(euler2)
    eulers1 = np.stack(eulers1, axis=0)
    eulers2 = np.stack(eulers2, axis=0)
    r_mse = np.mean((eulers1 - eulers2)**2, axis=-1)
    r_mae = np.mean(np.abs(eulers1 - eulers2), axis=-1)
    return r_mse, r_mae


def anisotropic_t_error(t1, t2):
    '''
    calculate translation mse and mae error.
    :param t1: shape=(B, 3)
    :param t2: shape=(B, 3)
    :return:
    '''
    if isinstance(t1, torch.Tensor):
        t1 = t1.cpu().detach().numpy()
    if isinstance(t2, torch.Tensor):
        t2 = t2.cpu().detach().numpy()
    assert t1.shape == t2.shape
    t_mse = np.mean((t1 - t2) ** 2, axis=1)
    t_mae = np.mean(np.abs(t1 - t2), axis=1)
    return t_mse, t_mae


def isotropic_R_error(r1, r2):
    '''
    Calculate isotropic rotation degree error between r1 and r2.
    :param r1: shape=(B, 3, 3), pred
    :param r2: shape=(B, 3, 3), gt
    :return:
    '''
    r2_inv = r2.permute(0, 2, 1).contiguous()
    r1r2 = torch.matmul(r2_inv, r1)
    # device = r1.device
    # B = r1.shape[0]
    # mask = torch.unsqueeze(torch.eye(3).to(device), dim=0).repeat(B, 1, 1)
    # tr = torch.sum(torch.reshape(mask * r1r2, (B, 9)), dim=-1)
    tr = r1r2[:, 0, 0] + r1r2[:, 1, 1] + r1r2[:, 2, 2]
    rads = torch.acos(torch.clamp((tr - 1) / 2, -1, 1))
    degrees = rads / math.pi * 180
    return degrees


def isotropic_t_error(t1, t2, R2):
    '''
    Calculate isotropic translation error between t1 and t2.
    :param t1: shape=(B, 3), pred_t
    :param t2: shape=(B, 3), gtt
    :param R2: shape=(B, 3, 3), gtR
    :return:
    '''
    R2, t2 = inv_R_t(R2, t2)
    error = torch.squeeze(R2 @ t1[..., None], -1) + t2
    error = torch.norm(error, dim=-1)
    return error