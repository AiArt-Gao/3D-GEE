import torch.nn as nn
import torch.nn.functional as F
import torch
import functools
from torchvision import models
from torch.autograd import Variable
import numpy as np
import math
from torchvision.models import vgg19
import adaattn


class Vgg_net(nn.Module):

    def __init__(self,sigmoid=True):
        # init vgg encoder
        image_encoder = nn.Sequential(
            nn.Conv2d(3, 3, (1, 1)),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(3, 64, (3, 3)),
            nn.ReLU(),  # relu1-1    torch.Size([1, 64, 256, 256])

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, (3, 3)),
            nn.ReLU(),  # relu1-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(),  # relu2-1     torch.Size([1, 128, 128, 128])

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, (3, 3)),
            nn.ReLU(),  # relu2-2
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(),  # relu3-1    torch.Size([1, 256, 64, 64])

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, (3, 3)),
            nn.ReLU(),  # relu3-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(),  # relu4-1, this is the last layer used   torch.Size([1, 512, 32, 32])

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu4-4
            nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-1      torch.Size([1, 512, 16, 16])

            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-2
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU(),  # relu5-3
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 512, (3, 3)),
            nn.ReLU()  # relu5-4
        )
        image_encoder.load_state_dict(torch.load('/home/yifan/yf/AdaAttN-main/vggmodule/vgg_normalised.pth'))
        enc_layers = list(image_encoder.children())
        # enc_1 = nn.DataParallel(nn.Sequential(*enc_layers[:4]).to(1), [0,1])
        # enc_2 = nn.DataParallel(nn.Sequential(*enc_layers[4:11]).to(1), [0,1])
        # enc_3 = nn.DataParallel(nn.Sequential(*enc_layers[11:18]).to(1), [0,1])
        # enc_4 = nn.DataParallel(nn.Sequential(*enc_layers[18:31]).to(1), [0,1])
        # enc_5 = nn.DataParallel(nn.Sequential(*enc_layers[31:44]).to(1), [0,1])

        enc_1 = nn.Sequential(*enc_layers[:4]).cuda()
        enc_2 = nn.Sequential(*enc_layers[4:11]).cuda()
        enc_3 = nn.Sequential(*enc_layers[11:18]).cuda()
        enc_4 = nn.Sequential(*enc_layers[18:31]).cuda()
        enc_5 = nn.Sequential(*enc_layers[31:44]).cuda()
        # self.mapnet = MappingNetwork(512,128,64)
        self.image_encoder_layers = [enc_1, enc_2, enc_3, enc_4, enc_5]
        for layer in self.image_encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
    def encode_with_intermediate(self, input_img):
        results = [input_img]
        for i in range(5):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    def encode_with_intermediate_level(self, input_img , level):
        results = [input_img]
        for i in range(5):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[level-1]

    def feature_normalize(self, feature_in, eps=1e-10):
        feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + eps
        feature_in_norm = torch.div(feature_in, feature_in_norm)
        return feature_in_norm

if __name__ == '__main__':
    CUDA_VISIBLE_DEVICES = '3'
    a = Vgg_net()
    b = torch.rand((3,3,256,256)).to(3)
    b = a.encode_with_intermediate_level(b,1)
    print(b.shape)
