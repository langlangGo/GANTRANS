import os
import numpy as np
from glob import glob
from PIL import Image
import torch
from torchvision.transforms import Compose, ToTensor, Resize
# from tifffile import imread
import imageio

def readData(root, train=True):
    data = []

    # 'gcs/data/segmentation_v2/train/groundTruth/1/1_00066_sub0_0_region1_10_Pos.png'
    if train:
        dir = os.path.join(root, 'train')
    else:
        dir = os.path.join(root, 'test')

    for i in ['1','2','3']:
        
        imagefNames = os.listdir(os.path.join(os.path.join(dir,'img'),i))
        #print(imagefNames[1:3])
        for imgfName in imagefNames:
            data.append([os.path.join(os.path.join(os.path.join(dir,'img'),i), imgfName),
                        os.path.join(os.path.join(os.path.join(dir,'groundTruth'),i), imgfName)])

    #print(data)
    return data



MIN_BOUND = 0
MAX_BOUND = 255


def normalize(image):
    """
    Perform standardization/normalization, i.e. zero_centering and setting
    the data to unit variance.
    """
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    return image

def change_gt(gt):
    c0 = np.copy(gt)
    # c1 = np.copy(gt)

    c0[c0 == 155] = 0
    c0[c0 == 255] = 1

    return c0


class LITS(torch.utils.data.Dataset):

    def __init__(self, root, transform=None, train=True):
        self.train = train
        self.root = root

        if not os.path.exists(self.root):
            raise Exception("[!] The directory {} does not exist.".format(root))

        self.paths = readData(root, self.train)




    def __getitem__(self, index):
        self.imagePath, self.maskPath = self.paths[index]
        image = imageio.v3.imread(self.imagePath)
        gt = imageio.v3.imread(self.maskPath)

        target = change_gt(gt)

        image = np.asanyarray(image)
        gt = np.asarray(target)

        return image, gt

    def __len__(self):
        return len(self.paths)


def loader(dataset, batch_size, num_workers=8, shuffle=True):

    input_images = dataset

    input_loader = torch.utils.data.DataLoader(dataset=input_images,
                                                batch_size=batch_size,
                                                shuffle=shuffle,
                                                num_workers=num_workers)

    return input_loader
