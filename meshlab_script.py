import pymeshlab
import os
from models.data import SimpleCadData
from config_utils import get_args
import glob

if __name__ == "__main__":
    params = get_args()
    base_path = 'D:\\data\\chairs\\03001627'
    ms = pymeshlab.MeshSet()
    names = []

    for filename in glob.iglob(params.modelsDataDir + '/*.mat'):
        names.append(filename)
    for i, name in enumerate(names[90:100]):
        print(os.path.join(base_path, name.split('\\')[-1].split('.')[0]))
