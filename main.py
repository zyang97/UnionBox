from config_utils import get_args
import torch
from models.network import Network
from losses import partComposition, tsdf_pred, chamfer_loss
from models.cuboid import CuboidSurface
from models.data import SimpleCadData
from torch.autograd import Variable
from generateCamera import load_file_name, read_img_list, read_proj_mat_list
import shutil
import os
import numpy as np
from tensorboardX import SummaryWriter

if True: # set output directory
    outputDir = 'output\\weights'
    if os.path.exists(outputDir):
        shutil.rmtree(outputDir)
    os.makedirs(outputDir)

    sampledPointsMeshDir = 'out\\sampled_points_mesh'
    if os.path.exists(sampledPointsMeshDir):
        shutil.rmtree(sampledPointsMeshDir)
    os.makedirs(sampledPointsMeshDir)

    outDirVal = 'out\\out_val'
    if os.path.exists(outDirVal):
        shutil.rmtree(outDirVal)
    os.makedirs(outDirVal)

    outDirTrain = 'out\\out_train'
    if os.path.exists(outDirTrain):
        shutil.rmtree(outDirTrain)
    os.makedirs(outDirTrain)

    viewOutDirVal = 'out\\out_view_val'
    if os.path.exists(viewOutDirVal):
        shutil.rmtree(viewOutDirVal)
    os.makedirs(viewOutDirVal)

    cuboidPointOutDirVal = 'out\\cuboidPointOut_val'
    if os.path.exists(cuboidPointOutDirVal):
        shutil.rmtree(cuboidPointOutDirVal)
    os.makedirs(cuboidPointOutDirVal)

    cuboidPointOutDirTrain = 'out\\cuboidPointOut_train'
    if os.path.exists(cuboidPointOutDirTrain):
        shutil.rmtree(cuboidPointOutDirTrain)
    os.makedirs(cuboidPointOutDirTrain)



params = get_args()

writer = SummaryWriter()
dataloader = SimpleCadData(params) # not implemented
network = Network(params).cuda()
optimizer = torch.optim.Adam(network.parameters(), lr=params.learningRate)
cuboidSampler = CuboidSurface(params.nSamplesChamfer, normFactor='Surf')
network.train()

# prev_params_featureNet = []
# for param in network.featureNet.parameters():
#     prev_params_featureNet.append(param)
#
# prev_params_predNet = []
# for param in network.predNet.parameters():
#     prev_params_predNet.append(param)

def train(imgs, projMatrices, dataloader, cuboidSampler, network, optimizer, nTrain, val=False):

    loss, coverage, consistency = 0, 0, 0
    i = 0
    writePts = False
    if val:
        i += nTrain
    for img, cam in zip(imgs, projMatrices):

        #print(i)

        sampledPointsMesh = dataloader.forward(i)
        sampledPointsMesh = Variable(sampledPointsMesh.cuda())
        # write sampled mesh points to file
        if not writePts:
            from models.boxes import OBJ
            obj = OBJ(sampledPointsMesh[0], [])
            obj.save_obj(os.path.join(sampledPointsMeshDir + '_' + str(i) + '.obj'))
            writePts = True

        _, predParts = network.forward(img, cam) # predParts: [B, N, 8]
        #predParts = predParts.view(1, params.numBoxes, 7) # predParts: [N, 7]
        optimizer.zero_grad()

        tsdf = tsdf_pred(sampledPointsMesh, predParts)
        coverageLossSub = tsdf.mean()

        chamfer, cuboidSampledPoints = chamfer_loss(predParts, dataloader, cuboidSampler)
        consistencyLossSub = chamfer.mean()

        lossSub = torch.sum(coverageLossSub + params.chamferLossWeight * consistencyLossSub)
        #loss = torch.mean(loss)
        #loss = torch.mean(loss)

        loss += lossSub
        coverage += coverageLossSub
        consistency += consistencyLossSub

        torch.cuda.empty_cache()
        i += 1

        # Loss.backward()
        # print("loss grad: ")
        # print(Loss.grad)
        # optimizer.step()
        # return loss.item(), coverageLoss.item(), consistencyLoss.item(), cuboidSampledPoints, predParts

    loss /= nTrain
    coverage /= nTrain
    consistency /= nTrain
    loss.backward()
    optimizer.step()
    return loss.item(), coverage.item(), consistency.item()


loss = 0
coverage = 0
consistency = 0

#names = load_file_name(params.camPath)
#imgs, projMatrices = read_img_list(params.imgPath, names), read_proj_mat_list(params.camPath, names)
#imgs = Variable(imgs.cuda())
#projMatrices = Variable(projMatrices.cuda())

imgTrain, camTrain, imgVal, camVal = dataloader.get_train_val_data()
nImgs = len(imgTrain)
imgTrain = Variable(imgTrain.cuda())
imgVal = Variable(imgVal.cuda())
camTrain = Variable(camTrain.cuda())
camVal = Variable(camVal.cuda())
names_val = dataloader.get_val_data_name()
names_train = dataloader.get_train_data_name()

for iter in range(params.numTrainIter):
    print("iter:{}\tLoss:{:10.7f}\tCoverage Loss:{:10.7f}\tConsistency Loss:{:10.7f}".format(int(iter), loss, coverage, consistency))

    # loss, coverage, consistency, cuboidSampledPoints, preds = train(imgTrain, camTrain, dataloader, cuboidSampler, network, optimizer, iter)
    loss, coverage, consistency = train(imgTrain, camTrain, dataloader, cuboidSampler, network, optimizer, nImgs)
    writer.add_scalar("Loss/train_loss", loss, iter)
    writer.add_scalar("Loss/train_coverage", coverage, iter)
    writer.add_scalar("Loss/train_consistency", consistency, iter)
    # netparams_featureNet = []
    # for param in network.featureNet.parameters():
    #     netparams_featureNet.append(param)
    #     #print(param.grad())
    # diff = np.sum([torch.sum(prev_params_featureNet[k] != netparams_featureNet[k]).cpu() for k in range(len(prev_params_featureNet))])
    # print('featureNet param diff is: {}'.format(diff))
    # prev_params_featureNet = netparams_featureNet.copy()
    #
    # netparams_predNet = []
    # for param in network.predNet.parameters():
    #     netparams_predNet.append(param)
    #     #print(param.grad())
    # diff = np.sum([torch.sum(prev_params_predNet[k] != netparams_predNet[k]).cpu() for k in range(len(prev_params_predNet))])
    # print('predNet param diff is: {}'.format(diff))
    # prev_params_predNet = netparams_predNet.copy()

    if iter % params.meshSaveIter == 0:

        for idx in range(5):
            network.eval()
            # _, preds = network.forward(imgs, projMatrices)
            _, preds = network.forward(imgTrain[idx], camTrain[idx])
            #preds = preds.view(1, params.numBoxes, 8)
            scores = preds[:, :, 0:1]
            cuboids = preds[:, :, 1:]
            mask = (scores > 0.5).squeeze()
            cuboids = cuboids.squeeze()[mask]
            cuboids = cuboids.view(1, cuboids.size(0), cuboids.size(1))
            network.train()

            # save boxes mesh
            from models.boxes import Boxes

            boxes = Boxes(cuboids)
            boxes.save_obj(os.path.join(outDirTrain, 'out_' + str(idx) + '_' + str(iter) + '.obj'))

            # save smapled points on boxes
            from losses import partComposition
            from models.boxes import OBJ

            sampledPoints, _ = partComposition(preds, cuboidSampler)
            obj = OBJ(sampledPoints[0], [])
            obj.save_obj(os.path.join(cuboidPointOutDirTrain, 'out_' + str(idx) + '_' + str(iter) + '.obj'))

        print('{}: Train object mesh and sampled points written.'.format(iter))


        for idx in range(10):
            network.eval()
            # _, preds = network.forward(imgs, projMatrices)
            _, preds = network.forward(imgVal[idx], camVal[idx])
            #preds = preds.view(1, params.numBoxes, 8)
            scores = preds[:, :, 0:1]
            cuboids = preds[:, :, 1:]
            mask = (scores > 0.5).squeeze()
            cuboids = cuboids.squeeze()[mask]
            cuboids = cuboids.view(1, cuboids.size(0), cuboids.size(1))
            network.train()

            # save boxes mesh
            from models.boxes import Boxes
            boxes = Boxes(cuboids)
            boxes.save_obj(os.path.join(outDirVal, 'out_' + str(idx) + '_' + str(iter) + '.obj'))

            # save smapled points on boxes
            from losses import partComposition
            from models.boxes import OBJ
            sampledPoints, _ = partComposition(preds, cuboidSampler)
            obj = OBJ(sampledPoints[0], [])
            obj.save_obj(os.path.join(cuboidPointOutDirVal, 'out_' + str(idx) + '_' + str(iter) + '.obj'))

        print('{}: Validation object mesh and sampled points written.'.format(iter))


        # from models.boxes import OBJ
        # obj = OBJ(cuboidSampledPoints[0], [])
        # obj.save_obj(os.path.join(cuboidPointOutDir, 'out_' + str(iter) + '.obj'))

    if iter % params.viewSaveIter == 0:

        for idx in range(10):
            network.eval()
            #features, _ = network.forward(imgs, projMatrices)
            features, _ = network.forward(imgVal[idx], camVal[idx])
            network.train()
            for i, viewPred in enumerate(features):
                viewPred = viewPred.view(1, params.numBoxes, 8)
                cuboids = viewPred[:, :, 1:]
                #boxes = Boxes(viewPred.cpu().detach().numpy())
                boxes = Boxes(cuboids)
                boxes.save_obj(os.path.join(viewOutDirVal, 'out_' + str(idx) + '_' + str(iter) + '_' + str(i+1) + '.obj'))
        print('{}: validation object views written.'.format(iter))

    if iter % params.ValIter == 0:
        network.eval()
        lossVal, coverageVal, consistencyVal = train(imgVal, camVal, dataloader, cuboidSampler, network, optimizer, nImgs, val=True)
        network.train()

        writer.add_scalar("Loss/val_loss", lossVal, iter)
        writer.add_scalar("Loss/val_coverage", coverageVal, iter)
        writer.add_scalar("Loss/val_consistency", consistencyVal, iter)

        torch.save(network.state_dict(), "{}/iter{}.pkl".format(params.snapshotDir, iter))
        print("iter:{}\tvalLoss:{:10.7f}\tvalCoverage Loss:{:10.7f}\tvalConsistency Loss:{:10.7f}".format(int(iter), lossVal, coverageVal, consistencyVal))


writer.close()