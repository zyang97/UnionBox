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
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, self.outputSize * 8, kernel_size=3, padding=1),
            #AnchorFlatten(self.outputSize)  # x, y, z, w, h, l, r
            nn.Flatten(),
            nn.Linear(38000, 3800), # 3800 for 8
            nn.Linear(3800, self.outputSize * 8),
        )


    def forward(self, x):
        features = self.core(x)
        preds = self.regressor(features)
        return preds

class PredNet(nn.Module):
    def __init__(self, params):
        super(PredNet, self).__init__()
        self.numBoxes = params.numBoxes
        self.inputSize = params.numViews * params.numBoxes * 8
        self.outputSize = params.numBoxes * 8 # score, x, y, z, h, w, l, r
        # init fc layers
        self.fcLayers = nn.Sequential(
            nn.Flatten(start_dim=0),
            nn.Linear(self.inputSize, self.outputSize),
            nn.Linear(self.outputSize, self.outputSize),
        )
        self.predLayer = nn.Sequential(
            nn.Linear(self.outputSize, params.numBoxes),
            nn.Sigmoid()
        )
        self.attriLayer = nn.Sequential(
            nn.Linear(self.outputSize, params.numBoxes*7),
        )

        # self.fcLayers.apply(weightsInit)


    def forward(self, x):
        x = self.fcLayers(x)
        attri = self.attriLayer(x)
        prob = self.predLayer(x)
        preds = torch.cat([prob, attri]).view(8, self.numBoxes).permute(1, 0)
        return preds

class Network(nn.Module):
    def __init__(self, params):
        #self.num_views = params.num_views
        super(Network, self).__init__()
        self.numBoxes = params.numBoxes
        self.featureNet = FeatureNet(params)
        self.predNet = PredNet(params)

    def forward(self, imgs, projMatrices):
        assert len(imgs) == len(projMatrices), "Different number of images and projection matrices"
        img_width, img_height = imgs.shape[2], imgs.shape[3]

        # step 1. predict boxes under camera coordinate on each single view
        preds = self.featureNet(imgs)

        # step 2. project boxes onto world coordinate
        features = homo_warping(preds, projMatrices) # features: [6, 80]

        # Step 3. feed into fc layers
        preds = self.predNet(features)
        preds = preds.unsqueeze(0)
        #preds = preds.view(1, self.numBoxes, 8)
        # scores = preds[:, :, 0:1]
        # preds = preds[:, :, 1:]
        # mask = (scores > 0.5).squeeze()
        # scores = scores.squeeze()[mask]
        # preds = preds.squeeze()[mask]

        return features, preds

