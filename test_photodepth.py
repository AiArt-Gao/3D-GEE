#!/usr/bin/python3

import argparse
import itertools

import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from model import Generator, GlobalGenerator2, InceptionV3
# from utils import ReplayBuffer
from utils import LambdaLR
from utils import channel2width
from utils import weights_init_normal
from utils import createNRandompatches
from dataset import UnpairedDepthDataset
import utils_pl
from collections import OrderedDict
import util.util as util
import networks
import numpy as np
import os
import glob

# ----------------------------   define Generator ---------------------------------#

checkpoint_dir = '/home/yifan/yf/informative-drawings-main/checkpoints/feats2Geom/feats2depth.pth'
netG = GlobalGenerator2(768, 3, n_downsampling=1, n_UPsampling=3)
netG.load_state_dict(torch.load(checkpoint_dir))
netG.eval()
netG.cuda()

net_recog = InceptionV3(55, False, use_aux=True, pretrain=True, freeze=True, every_feat=True)
net_recog.cuda()
net_recog.eval()

# ------------------------------  define dataset  ----------------------------------#

data_dir ='/data2/yifan/mjt/ssart/content'
image_list = glob.glob(data_dir + '/*.*')
depth_list = os.listdir('/data2/yifan/mjt/ssart/depthnew')
for i in image_list:
    n = i.split('/')[-1]
    print(n)
    print(i)
    if n in depth_list:
        pass
    else:
        img = Image.open(i)
        img = transforms.ToTensor()(img)
        img = img.unsqueeze(0).cuda()
        _,img = net_recog(img)
        output = netG(img)
        output = (output + 1)/2.0
        output = output.squeeze(0)
        output = transforms.ToPILImage()(output)
        img_name = os.path.basename(i)
        output.save('/data2/yifan/mjt/ssart/depthnew/' + img_name)




