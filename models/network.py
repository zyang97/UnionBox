import torch
from torch import nn
from models.resnet import resnet
from torch.utils.data import DataLoader
from models.utils import *


class AnchorFlatten(nn.Module):
  """
      Module for anchor-based network outputs,
      Init args:
          num_output: number of output channel for each anchor.

      Forward args:
          x: torch.tensor of shape [B, num_anchors * output_channel, H, W]

      Forward return:
          x : torch.tensor of shape [B, num_anchors * H * W, output_channel]
  """

  def __init__(self, num_output_channel):
    super(AnchorFlatten, self).__init__()
    self.num_output_channel = num_output_channel

  def forward(self, x):
    x = x.permute(0, 2, 3, 1)
    x = x.contiguous().view(x.shape[0], -1, self.num_output_channel)
    return x

class FeatureNet(nn.Module):
    def __init__(self, params):
        super(FeatureNet, self).__init__()
        self.numViews = params.numViews
        self.outputSize = params.numBoxes
        # init layer
        self.core = resnet(depth=101)
        self.regressor = nn.Sequential(
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, 2048, kernel_size=3, padding=1),
            nn.BatchNorm2d(2048),
            nn.ReLU(inplace=True),
            nn.Conv2d(2048, self.outputSize * 7, kernel_size=3, padding=1),
            #AnchorFlatten(self.outputSize)  # x, y, z, w, h, l, r
            nn.Flatten(),
            nn.Linear(33250, 3325), # 3325 for 7
            nn.Linear(3325, self.outputSize * 7),
        )


    def forward(self, x):
        features = self.core(x)
        preds = self.regressor(features)
        return preds

class PredNet(nn.Module):
    def __init__(self, params):
        super(PredNet, self).__init__()
        self.inputSize = params.numViews * params.numBoxes * 7
        self.outputSize = params.numBoxes * 7 # x, y, z, h, w, l, r, p
        # init fc layers
        self.fcLayers = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(self.inputSize, self.outputSize),
            nn.Linear(self.outputSize, self.outputSize),
        )

        # self.fcLayers.apply(weightsInit)


    def forward(self, x):
        preds = self.fcLayers(x)
        return preds


class Network(nn.Module):
    def __init__(self, params):
        #self.num_views = params.num_views
        super(Network, self).__init__()
        self.featureNet = FeatureNet(params)
        self.predNet = PredNet(params)

    def forward(self, imgs, projMatrices):
        assert len(imgs) == len(projMatrices), "Different number of images and projection matrices"
        img_width, img_height = imgs.shape[2], imgs.shape[3]

        # step 1. predict boxes under camera coordinate on each single view
        preds = self.featureNet(imgs)

        # step 2. project boxes onto world coordinate
        features = homo_warping(preds, projMatrices)

        # Step 3. feed into fc layers
        preds = self.predNet(features)

        return features, preds

