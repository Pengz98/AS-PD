import torch
from pcrnet.ops.transform_functions import PCRNetTransform as transform


def spam(pose_7d, template, source):
    batch_size = source.size(0)
    est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()  # (Bx3x3)
    est_t = torch.zeros(1, 3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()  # (Bx1x3)
    pose_7d = transform.create_pose_7d(pose_7d)

    # Find current rotation and translation.
    identity = torch.eye(3).to(source).view(1, 3, 3).expand(batch_size, 3, 3).contiguous()
    est_R_temp = transform.quaternion_rotate(identity, pose_7d).permute(0, 2, 1)
    est_t_temp = transform.get_translation(pose_7d).view(-1, 1, 3)

    # update translation matrix.
    est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
    # update rotation matrix.
    est_R = torch.bmm(est_R_temp, est_R)

    transform_source = transform.quaternion_transform(source, pose_7d)  # Ps' = est_R*Ps + est_t
    return est_R, est_t, transform_source