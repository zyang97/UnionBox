import torch
import math
from scipy.spatial.transform import Rotation as R

def translate(points, trans):
  nP = points.size(1)
  trans_rep = trans.repeat(1, nP, 1)
  return points - trans_rep

def euler_to_matrix(euler):
    theta1, theta2, theta3 = euler
    c1 = torch.cos(theta1 * torch.pi / 180)
    s1 = torch.sin(theta1 * torch.pi / 180)
    c2 = torch.cos(theta2 * torch.pi / 180)
    s2 = torch.sin(theta2 * torch.pi / 180)
    c3 = torch.cos(theta3 * torch.pi / 180)
    s3 = torch.sin(theta3 * torch.pi / 180)

    matrix = torch.tensor([[c2 * c3, -c2 * s3, s2],
                           [c1 * s3 + c3 * s1 * s2, c1 * c3 - s1 * s2 * s3, -c2 * s1],
                           [s1 * s3 - c1 * c3 * s2, c3 * s1 + c1 * s2 * s3, c1 * c2]], device='cuda:0')
    return matrix

def rotate(points, rot):
    rot = rot
    nP = points.size(1)
    # quat_rep = rot.repeat(1, nP, 1)
    # rotR = R.from_euler('xyz', [0, rot, 0], degrees=True)
    # rot_mat = torch.tensor(rotR.as_matrix(), dtype=torch.double).cuda()
    rot_mat = euler_to_matrix(torch.tensor([0, rot, 0], dtype=torch.float32, device='cuda:0'))
    rotated_points = torch.matmul(points, rot_mat)
    return rotated_points


def point_transform_world_to_local(sampledPoints, trans, rot):
    """
    transform sampled points from world coordinate into cuboids' coordination
    :param sampledPoints: [B, nPoints, 3]
    :param trans: [B, 1, 3]
    :param rot: [B, 1, 1]
    :return: [B, nPoints, 3]
    """
    trans_points = translate(sampledPoints, trans)
    rotated_points = rotate(trans_points, rot)
    return rotated_points

def point_transform_local_to_world(sampledPoints, trans, rot):
    """
    transform sampled points from local cuboids' coordinate into world coordination
    :param sampledPoints: [B, nPoints, 3]
    :param trans: [B, 1, 3]
    :param rot: [B, 1, 1]
    :return: [B, nPoints, 3]
    """
    rotated_points = rotate(sampledPoints, -rot)
    trans_points = translate(rotated_points, -trans)
    return trans_points

if __name__ == '__main__':
    # # test rotate
    # point = torch.tensor([[1, 1, 1], [2, 2, 2]], dtype=torch.double)
    # rotR = R.from_euler('xyz', [0, 0, 45], degrees=True)
    # rot_mat = torch.tensor(rotR.as_matrix(), dtype=torch.double)
    # print(torch.matmul(point, rot_mat))

    # # test transform
    # point = torch.tensor([[[1, 1, 1], [2, 2, 2]]], dtype=torch.double)
    # trans = torch.tensor([[[5, 5, 5]]])
    # rot = torch.tensor([[[45]]])
    # print(point_transform_world_to_local(point, trans, rot))
    # point2 = torch.tensor([[[-math.sqrt(32), 0.0, -4.0], [-math.sqrt(18), 0.0, -3.0]]], dtype=torch.double)
    # print(point_transform_local_to_world(point2, trans, rot))

    # test euler to matrix
    rot = torch.tensor([0, 12345, 0], dtype=torch.float32, device='cuda:0')
    rotR = R.from_euler('xyz', rot.cpu(), degrees=True)
    rot_mat = torch.tensor(rotR.as_matrix(), dtype=torch.double).cuda()
    rot_mat2 = euler_to_matrix(rot)
    print(rot_mat)
    print(rot_mat2)


"""
  [Parameter containing:
tensor([[-0.0121, -0.0363,  0.0302,  ..., -0.0466,  0.0272,  0.0025],
        [ 0.0445, -0.0020, -0.0235,  ...,  0.0331, -0.0051, -0.0149],
        [-0.0458,  0.0266,  0.0299,  ...,  0.0227,  0.0217,  0.0358],
        ...,
        [-0.0475,  0.0451, -0.0410,  ...,  0.0229, -0.0146,  0.0331],
        [ 0.0059, -0.0120,  0.0422,  ...,  0.0271, -0.0295,  0.0137],
        [ 0.0333,  0.0150,  0.0242,  ..., -0.0382,  0.0001, -0.0024]],
       device='cuda:0', requires_grad=True), Parameter containing:
tensor([ 0.0340, -0.0261, -0.0370, -0.0307, -0.0371, -0.0090, -0.0016,  0.0315,
         0.0324, -0.0014, -0.0384,  0.0201, -0.0237,  0.0185, -0.0096,  0.0318,
        -0.0230,  0.0191, -0.0022, -0.0211,  0.0122, -0.0124,  0.0274, -0.0257,
        -0.0185, -0.0437,  0.0487, -0.0364, -0.0224,  0.0439,  0.0510,  0.0384,
         0.0007,  0.0431, -0.0086, -0.0504, -0.0060, -0.0084, -0.0074,  0.0356,
         0.0061, -0.0158,  0.0447,  0.0174,  0.0055,  0.0218,  0.0141,  0.0112,
         0.0116, -0.0533,  0.0229,  0.0274,  0.0014,  0.0448, -0.0495, -0.0354,
        -0.0363,  0.0134, -0.0007,  0.0247,  0.0046, -0.0376,  0.0138, -0.0237,
         0.0328, -0.0353,  0.0137, -0.0321,  0.0232,  0.0273], device='cuda:0',
       requires_grad=True), Parameter containing:
tensor([[-0.0656, -0.0278, -0.0224,  ..., -0.0913,  0.0963,  0.1010],
        [-0.0613,  0.1081,  0.0446,  ...,  0.0178,  0.0991, -0.0112],
        [ 0.1050,  0.0363,  0.0858,  ...,  0.0959, -0.0190, -0.0413],
        ...,
        [ 0.0485, -0.0918, -0.0753,  ..., -0.0951, -0.0867,  0.0751],
        [ 0.0463, -0.0977, -0.0801,  ...,  0.0853,  0.0264,  0.0943],
        [ 0.0851,  0.0769,  0.0187,  ..., -0.1157,  0.0991,  0.0510]],
       device='cuda:0', requires_grad=True), Parameter containing:
tensor([ 0.0631, -0.0569,  0.0230,  0.0202,  0.0525,  0.0152, -0.0838,  0.0816,
         0.0914, -0.0429,  0.0055,  0.1178,  0.0278, -0.1059,  0.0830, -0.1072,
        -0.0524, -0.0015, -0.0754,  0.0678, -0.1020,  0.0836, -0.0748, -0.0640,
        -0.0195, -0.0061,  0.0839,  0.1079, -0.0291,  0.0015, -0.0270, -0.0023,
         0.0324, -0.0834,  0.0413,  0.0928,  0.0839,  0.0528,  0.0671,  0.1071,
        -0.0598,  0.0371,  0.0914,  0.0088, -0.0774, -0.1028,  0.0140, -0.0336,
         0.0077, -0.0992, -0.0015, -0.0613, -0.0537,  0.0154,  0.0739, -0.1151,
        -0.0008,  0.1098, -0.0784, -0.0047,  0.0231,  0.0485,  0.0615, -0.0268,
         0.0296, -0.0535,  0.0906, -0.0563, -0.0730, -0.0513], device='cuda:0',
       requires_grad=True)]
"""