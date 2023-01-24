import torch
# from emd import emd

def batch_pairwise_dist(x, y):
    bs, num_points_x, points_dim = x.size()
    _, num_points_y, _ = y.size()
    xx = x.pow(2).sum(dim=-1)
    yy = y.pow(2).sum(dim=-1)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand_as(zz.transpose(2, 1))
    ry = yy.unsqueeze(1).expand_as(zz)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P

def chamfer_dist(src, dst):
    # src: (B,P1,D), dst: (B,P2,D)
    src, dst = src.double(), dst.double()
    P = batch_pairwise_dist(dst, src)
    mins, _ = torch.min(P, 1)
    loss_1 = torch.mean(mins, 1)
    loss_1 = torch.mean(loss_1)
    mins, _ = torch.min(P, 2)
    loss_2 = torch.mean(mins, 1)
    loss_2 = torch.mean(loss_2)
    return loss_1 + loss_2

def hausdorff_dist(src, dst):
    src = src.double()
    dst = dst.double()
    distance_matrix = torch.cdist(src, dst, p=2)

    v1 = distance_matrix.min(2)[0].max(1, keepdim=True)[0]
    v2 = distance_matrix.min(1)[0].max(1, keepdim=True)[0]

    v = torch.cat([v1, v2], -1)
    return v.max(1)[0].mean()

# def earthmover_dist(src, dst):
#     src = src.cpu().data.numpy()
#     dst = dst.cpu().data.numpy()
#     dist = emd.emd(src, dst)
#     return dist


if __name__ == '__main__':
    src = torch.randn((4,128,3))
    dst = torch.randn((4,512,3))
    dist = earthmover_dist(src, dst)
    print(dist)