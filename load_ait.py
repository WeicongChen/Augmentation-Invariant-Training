'''Load images for Augmentation Invariant Training
'''
import torch
import torch.utils.data as data
from torch.utils.data.sampler import BatchSampler

from PIL import Image
import os
import random
import copy

def default_loader(path):
    img = Image.open(path).convert('RGB')
    assert img is not None, path
    return img

def default_list_reader(fileList):
    imgList = []
    labelList = []
    imgDict = {}
    with open(fileList, 'r') as file:
        for index, line in enumerate(file.readlines()):
            splits = line.strip().split()
            assert len(splits) == 2, 'wrong list ' + line
            name = splits[0]
            label = int(splits[1])
            imgList.append((name, label))
            assert '.' in name, index
            if label not in imgDict:
                imgDict[label] = [name]
            else:
                imgDict[label].append(name)
    return imgList, imgDict

class AITList(data.Dataset):
    def __init__(self, root, fileList, train=True, net_num=1, parallel=1, transforms=[], list_reader=default_list_reader, loader=default_loader):
        self.root = root
        self.imgList, imgDict = list_reader(fileList)
        self.transforms = transforms if isinstance(transforms, list) else [transforms]
        self.loader = loader
        self.net_num = net_num
        self.parallel = parallel
        self.train = train
        self.count = 0
        self.imgpath = ''
        self.tar = -1

    def __getitem__(self, index):
        if self.count == 0:
            imgpath, target = self.imgList[index // self.parallel]
            self.imgpath = imgpath
            self.tar = target
        self.count = (self.count + 1) % self.parallel
        img = self.loader(os.path.join(self.root, self.imgpath))
        imgs = []
        if len(self.transforms) != 0:
            if self.net_num == 1:
                return self.transforms[self.count](img), self.tar # for AIT and test
            for i in range(self.net_num):
                imgs.append(self.transforms[i](copy.copy(img))) # for multi-AIT
        return imgs, self.tar

    def __len__(self):
        return len(self.imgList) * self.parallel


