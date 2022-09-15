import argparse



def get_args():
    parser = argparse.ArgumentParser(description="Union Boxes")
    # Network
    parser.add_argument('--numViews', type=int, default=6)
    parser.add_argument('--numBoxes', type=int, default=10)
    # HyperParameters
    parser.add_argument('--learningRate', type=float, default=0.001)
    parser.add_argument('--numTrainIter', type=int, default=100000)
    parser.add_argument('--ValIter', type=int, default=10)
    parser.add_argument('--meshSaveIter', type=int, default=10)
    parser.add_argument('--viewSaveIter', type=int, default=100)
    parser.add_argument('--batchSize', type=int, default=1)
    parser.add_argument('--gridSize', type=int, default=32)
    parser.add_argument('--gridBound', type=float, default=0.5)
    parser.add_argument('--nSamplesChamfer', type=int, default=1500)
    parser.add_argument('--modelIter', type=int, default=2)
    parser.add_argument('--nSamplePoints', type=int, default=10000)
    parser.add_argument('--chamferLossWeight', type=int, default=1)
    # Path
    parser.add_argument('--snapshotDir', type=str, default='D:\\projects\\UnionBox2\\output\\weights')
    parser.add_argument('--imgPath', type=str, default='D:\\projects\\UnionBox2\\test2\\imgs')
    parser.add_argument('--camPath', type=str, default='D:\\projects\\UnionBox2\\test2\\cameraParam')
    parser.add_argument('--modelsDataDir', type=str, default="D:\\projects\\volumetricPrimitivesPytorch\\cachedir\\shapenet\\chamferData\\03001627")


    config = parser.parse_args()
    return config