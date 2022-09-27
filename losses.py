import torch
from projectUtils import *

import sys
from torch.autograd import Variable
from torch.nn import functional as F
import pdb


def tsdf_transform(sampledPoints, part):
    """
    :param sampledPoints: [B, nPoints, 3]
    :param part: [B, 1, 7]
    :return: []
    """
    trans = part[:, :, 0:3]  # B x 1 x 3
    shape = part[:, :, 3:6]  # B  x 1 x 3
    rot = part[:, :, 6:7]  # B x 1 x 1

    pointTransformed = point_transform_world_to_local(sampledPoints, trans, rot)

    nP = pointTransformed.size(1)
    shapeRep = shape.repeat(1, nP, 1)
    tsdf = torch.abs(pointTransformed) - shapeRep
    tsdf = F.relu(tsdf).pow(2).sum(dim=2)

    return tsdf

# def get_existence_weights(tsdf, part):
#   e = part[0,:,7:8]
#   e = e.expand(tsdf.size())
#   e = (1-e)*10
#   return e

def tsdf_pred(sampledPoints, predParts):
    """
    truncated signed distance function
    :param sampledPoints: [B, nPoints, 3]
    :param predParts: [B, nParts, 8]
    :return: [B, nPoints]
    """
    nParts = predParts.size(1)
    predParts = torch.chunk(predParts, nParts, dim=1)
    tsdfParts = []
    existenceWeights = []
    for i in range(nParts):
        tsdf = tsdf_transform(sampledPoints, predParts[i]) #
        tsdfParts.append(tsdf)
        #existenceWeights.append(get_existence_weights(tsdf, predParts[i]))

    #existenceAll = torch.cat(existenceWeights, dim=1)
    tsdfAll = torch.cat(tsdfParts, dim=1) #+ existenceAll
    tsdf_final = -1 * F.max_pool1d(-1 * tsdfAll, kernel_size=nParts)
    return tsdf_final

# def chamfer_transform(sampledPoints, part):
#     """
#     :param sampledPoints: [B, nPoints, 3]
#     :param part: [B, 1, 7]
#     :return: []
#     """
#     trans = part[:, :, 0:3]  # B x 1 x 3
#     #shape = part[:, :, 3:6]  # B  x 1 x 3
#     rot = part[:, :, 6:7]  # B x 1 x 1
#     pointTransformed = point_transform_local_to_world(sampledPoints, trans, rot)
#     return pointTransformed

# def primtive_surface_samples(predPart, cuboid_sampler):
#   # B x 1 x 10
#   shape = predPart[:, :, 3:6]  # B  x 1 x 3
#   # probs = predPart[:,:,11:12] # B x 1 x 1
#   samples = cuboid_sampler.sample_points_cuboid(shape)
#   #probs = probs.expand(imp_weights.size())
#   #imp_weights = imp_weights * probs
#   return samples #, imp_weights

def partComposition(predParts, cuboidSampler):
    """
    :param predParts: [B, nParts, 7]
    :param cuboidSampler:
    :return:
    """
    nParts = predParts.size(1)
    allSampledPoints = []
    #allSampledWeights = []
    predParts = torch.chunk(predParts, nParts, 1)
    for i in range(nParts):
        trans = predParts[i][:, :, 0:3]  # B x 1 x 3
        shape = predParts[i][:, :, 3:6]  # B  x 1 x 3
        rot = predParts[i][:, :, 6:7]  # B x 1 x 1
        #probs = predParts[i][:, :, 7:8]
        samplePoints = cuboidSampler.sample_points_cuboid(shape)
        transformedSamples = point_transform_local_to_world(samplePoints, trans, rot)
        #probs = probs.expand(sampleWeights.size())
        #sampleWeights = sampleWeights * probs
        allSampledPoints.append(transformedSamples)
        #allSampledWeights.append(sampleWeights)

    pointsOut = torch.cat(allSampledPoints, dim=1)
    #weightsOut = torch.cat(allSampledWeights, dim=1)
    return pointsOut #, weightsOut

def normalize_weights(imp_weights):
  # B x nP x 1
  totWeights = (torch.sum(imp_weights, dim=1) + 1E-6).repeat(1, imp_weights.size(1), 1)
  norm_weights = imp_weights / totWeights
  return norm_weights

def chamfer_loss(predParts, dataloader, cuboidSampler):
    sampledPointsCuboid = partComposition(predParts, cuboidSampler)
    #normWeights = normalize_weights(sampledWeights).squeeze()
    sampledPoints = sampledPointsCuboid.cuda()
    chamfer = dataloader.chamfer_forward(sampledPoints)
    # = chamfer * normWeights
    return chamfer, sampledPoints

def volume_pred(predParts):
    """
    Large boxes, high loss
    :param sampledPoints: [B, nPoints, 3]
    :return: [B, nPoints]
    """

if __name__ == '__main__':
    # # test tsdf
    # samplePoints = torch.tensor([[[3, 3, 3], [2, 2, 2], [1, 1, 1], [0, 0, 0]]], dtype=torch.double)
    # predParts = torch.tensor([[[0, 0, 0, 1, 1, 1, 0], [1, 1, 1, 1, 1, 1, 0]]])
    # #ret = torch.chunk(predParts, predParts.size(1), 1)
    # tsdf = tsdf_pred(samplePoints, predParts)
    # coverageLoss = tsdf.mean(dim=1).squeeze()
    # print(tsdf, coverageLoss)

    # # test chamfer
    # samplePoints = torch.tensor([[[3, 3, 3], [2, 2, 2], [1, 1, 1], [0, 0, 0]]], dtype=torch.double)
    # predParts = torch.tensor([[[0, 0, 0, 2, 4, 6, 0], [1, 1, 1, 1, 1, 1, 0]]], dtype=torch.double)
    # nParts = predParts.size(1)
    # predParts = torch.chunk(predParts, nParts, 1)
    # predPart = predParts[0]
    # shape = predPart[:, :, 3:6]
    # bs = shape.size(0)
    # ns = 9
    # nsp = ns // 3
    # dataType = shape.data.type()
    # coeffBernoulli = torch.bernoulli(torch.Tensor(bs, nsp, 3).fill_(0.5)).type(dataType)
    # coeffBernoulli = 2 * coeffBernoulli - 1  # makes entries -1 and 1
    #
    # coeff_w = torch.Tensor(bs, nsp, 3).type(dataType).uniform_(-1, 1)
    # coeff_w[:, :, 0].copy_(coeffBernoulli[:, :, 0].clone())
    #
    # coeff_h = torch.Tensor(bs, nsp, 3).type(dataType).uniform_(-1, 1)
    # coeff_h[:, :, 1].copy_(coeffBernoulli[:, :, 1].clone())
    #
    # coeff_d = torch.Tensor(bs, nsp, 3).type(dataType).uniform_(-1, 1)
    # coeff_d[:, :, 2].copy_(coeffBernoulli[:, :, 2].clone())
    #
    # coeff = torch.cat([coeff_w, coeff_h, coeff_d], dim=1)
    # coeff = Variable(coeff)
    #
    # dims = shape
    # dims_rep = dims.repeat(1, ns, 1)
    # res = dims_rep * coeff

    # test chamfer total

    from models.cuboid import CuboidSurface
    from config_utils import get_args
    from models.data import SimpleCadData

    params = get_args()
    dataloader = SimpleCadData(params)
    cuboidSampler = CuboidSurface(params.nSamplesChamfer, normFactor='Surf')
    predParts = torch.tensor([[[0, 0, 0, 2, 4, 6, 0], [1, 1, 1, 1, 1, 1, 0]]], dtype=torch.double, device='cuda:0')

    sampledPointsMesh = dataloader.forward()
    tsdf = tsdf_pred(sampledPointsMesh, predParts)
    coverageLoss = tsdf.mean(dim=1).squeeze().mean()


    sampledPointsCuboid = partComposition(predParts, cuboidSampler)
    sampledPoints = sampledPointsCuboid.cuda()
    chamfer = dataloader.chamfer_forward(sampledPoints)

    consistency = chamfer.mean()

    print(coverageLoss)
    print(consistency)