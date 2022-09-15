import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
import numpy as np

def weightsInit(m):
    name = str(type(m))
    if 'Conv3d' in name:
        m.weight.data = m.weight.data.normal_(mean=0,std=0.02)
        m.bias.data = m.bias.data.zero_()
    elif 'BatchNorm3d' in name:
        m.weight.data = m.weight.data.normal_(mean=1.0, std=0.02)
        m.bias.data = m.bias.data.zero_()

def homo_warping(fea, proj):
    """
    :param fea:   [B, 7 * N] /  [x, y, z, w, h, l, r]
    :param proj:  [B, 4 * 4]
    :return: [7 * N]
    """
    batch = fea.shape[0]
    N = fea.shape[1] // 7

    fea = fea.view(batch, N, 7) # [B, N, 7]

    with torch.no_grad():
        rot = proj[:, :3, :3] # [B, 3, 3]
        trans = proj[:, :3, :3:4] # [B, 3, 1]

        xyz = fea[:, :, :3]  # [B, N, 3]
        whl = fea[:, :, 3:6] # [B, N, 3]
        r = fea[:, :, 6:7] # [B, N, 1]
        #p = fea[:, :, 7:8] # [B, N, 1]

        # convert xyz
        #xyz = xyz.permute(0, 2, 1) # [B, 3, N]
        # xyz.contiguous()
        #rot_xyz = torch.matmul(rot, xyz) # [B, 3, N]
        #proj_xyz = rot_xyz + trans # [B, 3, N]
        xyz = torch.cat([xyz, torch.ones([batch, N, 1]).cuda()], dim=2)
        proj_xyz = []
        for xyz_, proj_ in zip(xyz, proj):
            projected = torch.matmul(xyz_, torch.linalg.inv(proj_))
            proj_xyz.append(projected[:,:3] / projected[:,3].unsqueeze(1))
        proj_xyz = torch.stack(proj_xyz)
        # proj_xyz = proj_xyz.permute(0, 2, 1)

        # convert r
        r = r.permute(0, 2, 1) # [B, 1, N]
        r.contiguous()
        rotR = R.from_matrix(rot.cpu())
        euler = rotR.as_euler("xyz", degrees=True)
        rotatedAngle = torch.tensor(euler[:,1], dtype=torch.float32, device='cuda:0').unsqueeze(1).unsqueeze(2).repeat(1, 1, N)
        world_r = r + rotatedAngle # convert rotation angle from camera coordinate to world coordinate

        # concat
        # proj_xyz = proj_xyz.permute(0, 2, 1) # [B, N, 3]
        # proj_xyz.contiguous()
        world_r = world_r.permute(0, 2, 1) # [B, N, 1]
        world_r.contiguous()
        proj_fea = torch.cat([proj_xyz, whl, world_r], dim=2) # [B, N, 8]

    proj_fea = proj_fea.view(batch, N*7)

    # with torch.no_grad():
    #     rot = proj[:3, :3] # [B, 3, 3]
    #     trans = proj[:3, :3:4] # [B, 3, 1]
    #
    #     xyz = fea[:, :3]  # [B, N, 3]
    #     whl = fea[:, 3:6] # [B, N, 3]
    #     r = fea[:, 6:7] # [B, N, 1]
    #
    #     # convert xyz
    #     xyz = xyz.permute(0, 2, 1) # [B, 3, N]
    #     xyz.contiguous()
    #     rot_xyz = torch.matmul(rot, xyz) # [B, 3, N]
    #     proj_xyz = rot_xyz + trans # [B, 3, N]
    #
    #     # convert r
    #     r = r.permute(0, 2, 1) # [B, 1, N]
    #     r.contiguous()
    #     rotR = R.from_matrix(rot)
    #     euler = rotR.as_euler("zxy", degrees=True)
    #     world_r = r + euler[0] # convert rotation angle from camera coordinate to world coordinate
    #
    #     # concat
    #     proj_xyz = proj_xyz.permute(0, 2, 1) # [B, N, 3]
    #     proj_xyz.contiguous()
    #     world_r = world_r.permute(0, 2, 1) # [B, N, 1]
    #     world_r.contiguous()
    #     proj_fea = torch.cat([proj_xyz, whl, world_r], dim=2) # [B, N, 7]
    # proj_fea = proj_fea.view(N * 7)

    return proj_fea


if __name__ == "__main__":
    pt = torch.tensor([-0.4287,  0.7006,  0.4808], dtype=torch.float32).cuda()
    pt2 = torch.tensor([-0.4287, 0.7006, 0.4808, 1], dtype=torch.float32).cuda()
    proj = torch.tensor([[0.9308, -0.0061, -0.3654, -0.7014],
                         [-0.1466, 0.9097, -0.3886, -0.8701],
                         [0.3348, 0.4153, 0.8458, -1.7721],
                         [0.0000, 0.0000, 0.0000, 1.0000]], dtype=torch.float32).cuda()
    #proj = torch.linalg.inv(proj)
    rot = proj[:3, :3]  # [B, 3, 3]
    trans = proj[:3, 3].squeeze()
    ptWorld = torch.matmul(pt-trans, torch.linalg.inv(rot))# [B, 3, 1]
    #ptWorld -= trans
    ptRes = torch.matmul(pt2, torch.linalg.inv(proj))
    ptRes2 = ptRes / ptRes[3]
    print(ptWorld)
    #res = homo_warping(torch.tensor([[-0.286294, 0.253813, 0.273985, 0, 0, 0, 0]]).cuda(), proj)
    #print(res)


# p = torch.tensor([[[[5, 10], [1, 2]]]])
# depth_values = torch.tensor([[4]])
# depth_values = depth_values.view(*depth_values.shape, 1, 1)
# depth = torch.sum(p * depth_values, axis=1)
