import os
import torch
import math
from scipy.spatial.transform import Rotation as R
from xml.dom import minidom
import numpy as np
import cv2
def load_file_name(source):
    files = os.listdir(source)
    fileNames = []
    for file in files:
        fileNames.append(file.split('.')[0])
    return fileNames

def read_image(source, name):
    img = cv2.imread(os.path.join(source, name))
    #print(source, name)
    img = cv2.resize(img, (800, 600), interpolation = cv2.INTER_AREA)
    # cv2.imshow('img', img)
    # cv2.waitKey(0)
    return img

def read_proj_mat(source, name):
    file = minidom.parse(os.path.join(source, name))
    model = file.getElementsByTagName('project')
    trans = model[0].getElementsByTagName('VCGCamera')[0].attributes['TranslationVector'].value.split(' ')
    trans = np.array([float(x) for x in trans])
    rotMat = model[0].getElementsByTagName('VCGCamera')[0].attributes['RotationMatrix'].value.strip().split(' ')
    rotMat = np.array([float(x) for x in rotMat]).reshape((4, 4))
    rotMat[:3,3] = trans[:3]
    return rotMat

def read_img_list(source, names):
    print("Load imgs...")
    imgs = []
    for name in names:
        imgsMesh = []
        absPath = os.path.join(source, name)
        li = os.listdir(absPath)
        for fileName in li:
            imgsMesh.append(read_image(absPath, fileName))
        imgs.append(imgsMesh)
        #imgs.append(read_image(source, name))
    print("Done!")
    return torch.tensor(np.array(imgs), dtype=torch.float32).permute(0, 1, 4, 3, 2)
    #return np.transpose(np.array(imgs), (0, 3, 1, 2))

def read_proj_mat_list(source, names):
    print("Load camera calibrations...")
    cams = []
    for name in names:
        camsMesh = []
        absPath = os.path.join(source, name)
        li = os.listdir(absPath)
        for fileName in li:
            camsMesh.append(read_proj_mat(absPath, fileName))
        cams.append(camsMesh)
    print("Done!")
    return torch.tensor(np.array(cams), dtype=torch.float32)
    #return np.array(projMats)

if __name__ == '__main__':

    fileNames = load_file_name("D:\\projects\\UnionBox2\\test\\cameraParam")
    imgs, projMats = [], []

    img_source = 'D:\\projects\\UnionBox2\\test\\imgs'
    xml_source = "D:\\projects\\UnionBox2\\test\\cameraParam"

    imgs = read_img_list(img_source, fileNames)
    projMats = read_proj_mat_list(xml_source, fileNames)

    proj = projMats[0]

    rot = proj[:3, :3]  # [B, 3, 3]
    trans = proj[:3, :3:4]  # [B, 3, 1]

    pt = torch.tensor([-0.286294, 0.253813, 0.273985, 1], dtype=torch.float32)
    transPt = torch.matmul(pt, proj)
    transPt2 = transPt / transPt[3]
    ptRev = torch.matmul(transPt2, torch.linalg.inv(proj))
    ptRev2 = ptRev / ptRev[3]

    print(imgs)
    print(projMats)

