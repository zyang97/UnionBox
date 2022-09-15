import torch
import torch.nn as nn
from torch.autograd import Variable

class CuboidSurface(nn.Module):
    def __init__(self, nSamples, normFactor='None'):
        self.nSamples = nSamples
        self.samplesPerFace = nSamples // 3
        self.normFactor = normFactor

    def cuboidAreaModule(self, dims):
        width, height, depth = torch.chunk(dims, chunks=3, dim=2)

        wh = width * height
        hd = height * depth
        wd = width * depth

        surfArea = 2 * (wh + hd + wd)
        areaRep = surfArea.repeat(1, self.nSamples, 1)
        return areaRep

    def sample_wt_module(self, dims):
        # dims is bs x 1 x 3
        area = self.cuboidAreaModule(dims)  # bs x 1 x 1
        dimsInv = dims.pow(-1)
        dimsInvNorm = dimsInv.sum(2).repeat(1, 1, 3)
        normWeights = 3 * (dimsInv / dimsInvNorm)

        widthWt, heightWt, depthWt = torch.chunk(normWeights, chunks=3, dim=2)
        widthWt = widthWt.repeat(1, self.samplesPerFace, 1)
        heightWt = heightWt.repeat(1, self.samplesPerFace, 1)
        depthWt = depthWt.repeat(1, self.samplesPerFace, 1)

        sampleWt = torch.cat([widthWt, heightWt, depthWt], dim=1)
        finalWt = (1 / self.samplesPerFace) * (sampleWt * area)
        return finalWt

    def sample_points_cuboid(self, shape):
        """
        :param shape: [B, 1, 3]
        :return: samples: [B, nSamples, 3]
        """
        batchSize = shape.size(0)
        samplesPerFace = self.samplesPerFace
        dataType = shape.data.type()

        coeffBernoulli = torch.bernoulli(torch.Tensor(batchSize, samplesPerFace, 3).fill_(0.5)).type(dataType)
        coeffBernoulli = (2 * coeffBernoulli - 1)  # makes entries -1 and 1

        coeff_w = torch.Tensor(batchSize, samplesPerFace, 3).type(dataType).uniform_(-1, 1)
        coeff_w[:, :, 0].copy_(coeffBernoulli[:, :, 0].clone())

        coeff_h = torch.Tensor(batchSize, samplesPerFace, 3).type(dataType).uniform_(-1, 1)
        coeff_h[:, :, 1].copy_(coeffBernoulli[:, :, 1].clone())

        coeff_d = torch.Tensor(batchSize, samplesPerFace, 3).type(dataType).uniform_(-1, 1)
        coeff_d[:, :, 2].copy_(coeffBernoulli[:, :, 2].clone())

        coeff = torch.cat([coeff_w, coeff_h, coeff_d], dim=1)
        coeff = Variable(coeff)

        shape_rep = shape.repeat(1, self.nSamples, 1)
        samples = shape_rep * coeff
        #weights = self.sample_wt_module(shape)

        return samples #, weights
