#!/usr/bin/python3
import os
import torchvision.transforms as transforms
import torch.nn.functional as F
import cv2
import argparse
import itertools
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch
from torch.nn import DataParallel
from model import Generator, GlobalGenerator2, InceptionV3 , MappingNetwork,Generator_line
# from utils import ReplayBuffer
from utils import LambdaLR
from utils import channel2width
from utils import weights_init_normal
from utils import createNRandompatches
from dataset import UnpairedDepthDataset,OneDataset
import utils_pl
import random
from collections import OrderedDict
import util.util as util
import networks
import numpy as np
from ops import dequeue_data, queue_data
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch.nn as nn
import adaattn
from  Vggnet  import *
from adaways import  *
from torchvision.utils import save_image
from depth_anything import test
from loss import *
from aesNceloss import *
from ROPE.rope import RotaryEmbedding
from hist_loss import RGBuvHistBlock
import matplotlib

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str,default='chinafinal',help='name of this experiment')
parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='Where checkpoints are saved')
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=45, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=4, help='size of the batches')
parser.add_argument('--cuda', default=True,help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
###loading data
parser.add_argument('--dataroot', type=str, default='/data2/yifan/mjt/ssart/content', help='photograph directory root directory')
parser.add_argument('--root2', type=str, default='/data2/yifan/mjt/ssart/style', help='line drawings dataset root directory')
parser.add_argument('--depthroot', type=str, default='/data2/yifan/mjt/ssart/depthnew', help='dataset of corresponding ground truth depth maps')
parser.add_argument('--feats2Geom_path', type=str, default='/home/yifan/yf/informative-drawings-adaattn/checkpoints/feats2depth.pth',
                                help='path to pretrained features to depth map network')

### architecture and optimizers
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum for optimizer')
parser.add_argument('--decay_epoch', type=int, default=5, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--geom_nc', type=int, default=3, help='number of channels of geom data')
parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
parser.add_argument('--netD', type=str, default='basic', help='selects model to use for netD')
parser.add_argument('--n_blocks', type=int, default=3, help='number of resnet blocks for generator')
parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
parser.add_argument('--disc_sigmoid', type=int, default=0, help='use sigmoid in disc loss')
parser.add_argument('--every_feat', type=int, default=1, help='use transfer features for recog loss')
parser.add_argument('--finetune_netGeom', type=int, default= 0, help='make geometry networks trainable')

### loading from checkpoints
parser.add_argument('--load_pretrain', type=str, default='', help='where to load file if wanted')
parser.add_argument('--continue_train', default=True,help='continue training: load the latest model')
parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load from if continue_train')

### dataset options
parser.add_argument('--mode', type=str, default='train', help='train, val, test, etc')
parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
parser.add_argument('--aspect_ratio', type=float, default=1.0, help='The ratio width/height. The final height of the load image will be crop_size/aspect_ratio')
parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
parser.add_argument('--contrastive_weight', type=int,default=0,help='use contrastive loss')
######## loss weights
parser.add_argument("--cond_cycle", type=float, default=0.1, help="weight of the appearance reconstruction loss")
parser.add_argument("--condGAN", type=float, default=1.0, help="weight of the adversarial style loss")
parser.add_argument("--cond_recog", type=float, default=10.0, help="weight of the semantic loss")
parser.add_argument("--condGeom", type=float, default=10.0, help="weight of the geometry style loss")

### geometry loss options
parser.add_argument('--use_geom', type=int, default=0, help='include the geometry loss')
parser.add_argument('--midas', type=int, default=0, help='use midas depth map')

### semantic loss options
parser.add_argument('--N_patches', type=int, default=1, help='number of patches for clip')
parser.add_argument('--patch_size', type=int, default=128, help='patchsize for clip')
parser.add_argument('--num_classes', type=int, default=55, help='number of classes for inception')
parser.add_argument('--cos_clip', type=int, default=0, help='use cosine similarity for CLIP semantic loss')

### save options
parser.add_argument('--save_epoch_freq', type=int, default=1, help='how often to save the latest model in steps')
parser.add_argument('--slow', type=int, default=0, help='only frequently save netG_A, netGeom')
parser.add_argument('--log_int', type=int, default=50, help='display frequency for tensorboard')

### adaattn parameters

parser.add_argument('--lambda_content', type=float, default=0., help='weight for L2 content loss')
parser.add_argument('--lambda_global', type=float, default=10., help='weight for L2 style loss')
parser.add_argument('--lambda_local', type=float, default=3.,help='weight for attention weighted style loss')
parser.add_argument('--skip_connection_3',default=True,help='if specified, add skip connection on ReLU-3')
parser.add_argument('--shallow_layer',default=True,help='if specified, also use features of shallow layers')
parser.add_argument('--init_type', type=str, default='normal',help='network initialization [normal | xavier | kaiming | orthogonal]')
parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')

opt = parser.parse_args()
print(opt)

checkpoints_dir = opt.checkpoints_dir 
name = opt.name

from util.visualizer2 import Visualizer
tensor2im = util.tensor2imv2



def calc_remd_loss(A, B):
    C = cosine_dismat(A, B)
    m1, _ = C.min(1)
    m2, _ = C.min(2)

    remd = torch.max(m1.mean(), m2.mean())

    return remd
def calc_histogram_loss(A, B, histogram_block):
    input_hist = histogram_block(A)
    target_hist = histogram_block(B)
    histogram_loss = (1/np.sqrt(2.0) * (torch.sqrt(torch.sum(
        torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) /
        input_hist.shape[0])

    return histogram_loss

def cosine_dismat(A, B):
    A = A.view(A.shape[0], A.shape[1], -1)
    B = B.view(B.shape[0], B.shape[1], -1)

    A_norm = torch.sqrt((A**2).sum(1))
    B_norm = torch.sqrt((B**2).sum(1))

    A = (A/A_norm.unsqueeze(dim=1).expand(A.shape)).permute(0,2,1)
    B = (B/B_norm.unsqueeze(dim=1).expand(B.shape))
    dismat = 1.-torch.bmm(A, B)

    return dismat
def calc_ss_loss(A, B):
    MA = cosine_dismat(A, A)
    MB = cosine_dismat(B, B)
    Lself_similarity = torch.abs(MA-MB).mean()

    return Lself_similarity
def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = torch.bmm(features,features_t)
    return gram


def calc_contrastive_loss(query, key, queue, temp=0.07):
    N = query.shape[0]
    K = queue.shape[0]

    zeros = torch.zeros(N, dtype=torch.long, device=query.device)
    key = key.detach()

    logit_pos = torch.bmm(query.view(N, 1, -1), key.view(N, -1, 1))
    logit_neg = torch.mm(query.view(N, -1), queue.t().view(-1, K))

    logit = torch.cat([logit_pos.view(N, 1), logit_neg], dim=1)

    loss = F.cross_entropy(logit / temp, zeros)

    return loss


visualizer = Visualizer(checkpoints_dir, name, tf_log=True, isTrain=True)
print('------------------- created visualizer -------------------')

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
# define attaada3 ,4 ,5
max_sample = 64 * 64
if opt.skip_connection_3:
    adaattn_3 = adaattn.AdaAttN(in_planes=256, key_planes=256 + 128 + 64 if opt.shallow_layer else 256,
                                max_sample=max_sample)
    net_adaattn_3 = adaattn.init_net(adaattn_3, opt.init_type, opt.init_gain)
if opt.shallow_layer:
    channels = 512 + 256 + 128 + 64
else:
    channels = 512
transformer = adaattn.Transformer(
    in_planes=512, key_planes=channels, shallow_layer=opt.shallow_layer)
net_transformer = adaattn.init_net(transformer, opt.init_type, opt.init_gain)

# define Vgg
vggnet = Vgg_net()


netG_A = 0
netG_A = adaattn.Decoder(opt.skip_connection_3)
netG_A = adaattn.init_net(netG_A, opt.init_type, opt.init_gain)
# netG_A = Generator(opt.input_nc, opt.output_nc,opt.n_blocks)
# netG_B = adaattn.Decoder(False)
# netG_B = adaattn.init_net(netG_B, opt.init_type, opt.init_gain)

netG_B = Generator(opt.output_nc, opt.input_nc,opt.n_blocks)
if opt.use_geom == 1:

    netGeom = GlobalGenerator2(768, opt.geom_nc, n_downsampling=1, n_UPsampling=3)
    netGeom.load_state_dict(torch.load(opt.feats2Geom_path))

    print("Loading pretrained features to depth network from %s"%opt.feats2Geom_path)
    if opt.finetune_netGeom == 0:
        netGeom.eval()
else:
    opt.finetune_netGeom = 0


D_input_nc_B = opt.output_nc
D_input_nc_A = opt.input_nc

netD_B = networks.define_D(D_input_nc_B, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid=False)
netD_A = networks.define_D(D_input_nc_A, opt.ndf, opt.netD, opt.n_layers_D, opt.norm, use_sigmoid=False)



device = 'cuda'
if opt.cuda:
    netG_A.cuda()
    netG_B.cuda()
    netD_A.cuda()
    netD_B.cuda()
    if opt.use_geom==1:
        netGeom.cuda()
    device = 'cuda'

### load pretrained inception
net_recog = InceptionV3(opt.num_classes, opt.mode=='test', use_aux=True, pretrain=True, freeze=True, every_feat=opt.every_feat==1)
net_recog.cuda()
net_recog.eval()

import clip
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
clip.model.convert_weights(clip_model)

#### load in progress weights if continue train or load_pretrain
if opt.continue_train:
    netG_A.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_A_%s.pth' % opt.which_epoch)))
    netG_B.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netG_B_%s.pth' % opt.which_epoch)))
    netD_A.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netD_A_%s.pth' % opt.which_epoch)))
    netD_B.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netD_B_%s.pth' % opt.which_epoch)))
    net_adaattn_3.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'adaattn3_%s.pth' % opt.which_epoch)))
    net_transformer.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'transformer_%s.pth' % opt.which_epoch)))
    if opt.finetune_netGeom == 1:
        netGeom.load_state_dict(torch.load(os.path.join(opt.checkpoints_dir, opt.name, 'netGeom_%s.pth'% opt.which_epoch)))
    print('----------- loaded %s from '%opt.which_epoch + os.path.join(checkpoints_dir, name) + '---------------------- !!')
elif len(opt.load_pretrain) > 0:
    pretrained_path = opt.load_pretrain
    netG_A.load_state_dict(torch.load(os.path.join(pretrained_path, 'netG_A_%s.pth' % opt.which_epoch)))
    netG_B.load_state_dict(torch.load(os.path.join(pretrained_path, 'netG_B_%s.pth' % opt.which_epoch)))
    netD_A.load_state_dict(torch.load(os.path.join(pretrained_path, 'netD_A_%s.pth' % opt.which_epoch)))
    netD_B.load_state_dict(torch.load(os.path.join(pretrained_path, 'netD_B_%s.pth' % opt.which_epoch)))
    if opt.finetune_netGeom == 1:
        netGeom.load_state_dict(torch.load(os.path.join(pretrained_path, 'netGeom_%s.pth'% opt.which_epoch)))
    print('----------- loaded %s from '%opt.which_epoch + ' ' + pretrained_path + '---------------------- !!')
else:
    netG_A.apply(weights_init_normal)
    netG_B.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)
    

print('----------- loaded networks ---------------------- !!')

# Losses

criterionGAN = networks.GANLoss(use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, reduceme=True).to(device)

criterion_MSE = torch.nn.MSELoss(reduce=True)
criterionCycle = torch.nn.L1Loss()
criterionCycleB = criterionCycle


criterionCLIP = criterion_MSE
if opt.cos_clip == 1:
    criterionCLIP = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

criterionGeom = torch.nn.BCELoss(reduce=True)


############### only use B to A ###########################
optimizer_G_A = torch.optim.Adam(netG_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_G_B = torch.optim.Adam(netG_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))

if opt.skip_connection_3:
    optim_atta_3 = torch.optim.Adam(net_adaattn_3.parameters(), lr=0.0002)
else:
    optim_atta_3 = None

optim_atta_45 = torch.optim.Adam(net_transformer.parameters(), lr=0.0002)

if (opt.use_geom == 1 and opt.finetune_netGeom==1):
    optimizer_Geom = torch.optim.Adam(netGeom.parameters(), lr=opt.lr, betas=(0.5, 0.999))


optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))

lr_scheduler_G_A = torch.optim.lr_scheduler.LambdaLR(optimizer_G_A,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

lr_scheduler_G_B = torch.optim.lr_scheduler.LambdaLR(optimizer_G_B,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A,
                                                     lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

if opt.skip_connection_3:
    lr_scheduler_attn3 = torch.optim.lr_scheduler.LambdaLR(optim_atta_3,lr_lambda=LambdaLR(opt.n_epochs, opt.epoch,opt.decay_epoch).step)
else:
    lr_scheduler_attn3 = None

lr_scheduler_attn45 = torch.optim.lr_scheduler.LambdaLR(optim_atta_45,
                                                   lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensorreal_A
cal_depth_object = test.cal_depth_loss()
# Dataset loader
transforms_r = [transforms.Resize(int(opt.size * 1.12), Image.BICUBIC),
               transforms.RandomCrop(opt.size),
               transforms.ToTensor()]
#
# train_data = UnpairedDepthDataset(opt.dataroot, opt.root2, opt, transforms_r=transforms_r,
#                 mode=opt.mode, midas=opt.midas>0, depthroot=opt.depthroot)

# dataloader = DataLoader(train_data, batch_size=opt.batchSize, shuffle=True,num_workers=opt.n_cpu, drop_last=True)

# net_G_line = 0
# net_G_line = Generator_line(opt.input_nc, 1, opt.n_blocks)
# net_G_line.cuda()
# net_G_line.load_state_dict(torch.load('/home/yifan/yf/informative-drawings-line/checkpoints/anime_style/netG_A_latest.pth'))
# net_G_line.eval()

# print('---------------- loaded %d images ----------------' % len(train_data))
hist = RGBuvHistBlock(insz=64, h=256,  intensity_scale=True,  method='inverse-quadratic', device=device)
# queue = torch.rand((1024, 4096), dtype=torch.float).cuda()
# train_data = UnpairedDepthDataset(opt.dataroot, opt.root2, opt, transforms_r=transforms_r,
#                                   mode=opt.mode, midas=opt.midas > 0, depthroot=opt.depthroot)
# cmap = matplotlib.colormaps.get_cmap('Spectral_r')
style_data = OneDataset(opt.root2, opt, transforms_r=transforms_r)
content_data = OneDataset(opt.dataroot, opt, transforms_r=transforms_r)
# rope = RotaryEmbedding(256,device)
dataloader_style = DataLoader(style_data, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)
dataloader_content = DataLoader(content_data, batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)
adaway = Ada_way(opt)
# patch_loss = Patch_loss()
# cal_aec_loss = cal_style_patch()
# avgpool = nn.AdaptiveAvgPool2d((1, 1))
###### Training ######

for epoch in range(opt.epoch, opt.n_epochs):
    for i, (batch1, batch2) in enumerate(zip(dataloader_style, dataloader_content)):
        total_steps = epoch*len(dataloader_style) + i

        img_r = Variable(batch2['r']).cuda()
        img_depth = Variable(batch2['depth']).cuda()
        real_A_input = img_r
        real_A = real_A_input.clone().detach()
        real_A_input = real_A_input
        real_B = 0
        real_B = Variable(batch1['r']).cuda()
        recover_geom = img_depth
        batch_size = real_A.size()[0]
        condGAN = opt.condGAN
        cond_recog = opt.cond_recog
        cond_cycle = opt.cond_cycle
        ######################  rope code embedding  #########################
        # depth_gene = run2.Depth_loss(device)
        # real_A_depth2 = depth_gene.get_depth(real_A)
        # real_A_depth2 = real_A_depth2.unsqueeze(1)
        # a1, b1 = rope.Create_patch_feature(real_A_depth2, 16)
        # real_rope = rope(a1, b1)
        # real_rope = real_rope.repeat(1, 3, 1, 1)
        # real_rope = real_rope.view(4, 3, 256, 16, 16)
        # real_rope = (real_rope - real_rope.min()) / (real_rope.max() - real_rope.min())
        # real_rope_zeros = torch.zeros(4,3,256,256).cuda()
        # for i in range(256):
        #     x = i % 16
        #     y = i // 16
        #     patch = real_rope[:, :, i, :, :]
        #     real_rope_zeros[:, :, x * 16:x * 16 + 16, y * 16: y * 16 + 16] = patch
        # real_rope_zeros = torch.transpose(real_rope_zeros, 2,3)
        # real_A = 0.1 * real_rope_zeros + 0.9 * real_A
        # real_A = (real_A - real_A.min()) / (real_A.max() - real_A.min())
        # #################### Generator ####################

        # real_A_feat = netG_A.get_content_feature(real_A)  # G_A(A)
        real_A_feat = vggnet.encode_with_intermediate(real_A)  # G_A(A)
        style_A_feat = vggnet.encode_with_intermediate(real_B)

        if opt.skip_connection_3:
            c_adain_feat_3 = net_adaattn_3(real_A_feat[2], style_A_feat[2],adaway.get_key(real_A_feat, 2, opt.shallow_layer),
                                           adaway.get_key(style_A_feat, 2, opt.shallow_layer), adaway.seed)
        else:
            c_adain_feat_3 = None

        cs = net_transformer(real_A_feat[3], style_A_feat[3], real_A_feat[4], style_A_feat[4],
                                  adaway.get_key(real_A_feat, 3, opt.shallow_layer),
                                  adaway.get_key(style_A_feat, 3, opt.shallow_layer),
                                  adaway.get_key(real_A_feat, 4, opt.shallow_layer),
                                  adaway.get_key(style_A_feat, 4, opt.shallow_layer), adaway.seed)

        # print(c_adain_feat_3.shape)
        # fake_B = netG_A(c3 = c_adain_feat_3, cs = cs, x = 0 , rec = False)
        # print(cs.shape)   # torch.Size([4, 512, 32, 32])
        # print(c_adain_feat_3.shape)   # torch.Size([4, 256, 64, 64])
        fake_B = netG_A(cs, c_adain_feat_3,cc = real_A_feat,epoch = epoch)
        fake_B_feat = vggnet.encode_with_intermediate(fake_B)
        rec_A = netG_B(c3=0, cs=0, x=fake_B, rec=True)  # G_B(G_A(A))
        real_B_feat = vggnet.encode_with_intermediate(real_B)
        fake_A = netG_B(c3 = 0, cs = 0, x = real_B,rec = True)   # G_B(B)
        #
        # fake_A_feat = netG_A.get_content_feature(fake_A)  # G_A(A)
        fake_A_feat = vggnet.encode_with_intermediate(fake_A)
        if opt.skip_connection_3:
            c_adain_feat_3_rec = net_adaattn_3(fake_A_feat[2], style_A_feat[2],
                                           adaway.get_key(fake_A_feat, 2, opt.shallow_layer),
                                           adaway.get_key(style_A_feat, 2, opt.shallow_layer), adaway.seed)
        else:
            c_adain_feat_3_rec = None

        cs_rec = net_transformer(fake_A_feat[3], style_A_feat[3], fake_A_feat[4], style_A_feat[4],
                             adaway.get_key(fake_A_feat, 3, opt.shallow_layer),
                             adaway.get_key(style_A_feat, 3, opt.shallow_layer),
                             adaway.get_key(fake_A_feat, 4, opt.shallow_layer),
                             adaway.get_key(style_A_feat, 4, opt.shallow_layer), adaway.seed)


        # rec_B = netG_A(c3 = c_adain_feat_3_rec, cs = cs_rec, x = 0 , rec = False) # G_A(G_B(B))
        rec_B = netG_A(cs_rec, c_adain_feat_3_rec , cc = fake_A_feat , epoch = epoch)

        output_feats = vggnet.encode_with_intermediate(fake_B)

        adaway.cs = output_feats
        adaway.s_feats  = style_A_feat
        adaway.c_feats  = real_A_feat
        adaway.compute_losses()
        loss_ada_local = adaway.loss_local
        loss_ada_gobal = adaway.loss_global
        loss_ada_content = adaway.loss_content

        loss_cycle_Geom = 0
        if opt.use_geom == 1:
            geom_input = fake_B
            if geom_input.size()[1] == 1:
                geom_input = geom_input.repeat(1, 3, 1, 1)
            _, geom_input = net_recog(geom_input)
            pred_geom = netGeom(geom_input)
            pred_geom = (pred_geom+1)/2.0 ###[-1, 1] ---> [0, 1]
            loss_cycle_Geom = criterionGeom(pred_geom, recover_geom)

            ##   PAMA  ####
        #
        # for j in range(2,5):
        #     loss_contentAMA = 5 * calc_ss_loss(real_A_feat[j],fake_B_feat[j])
        #     loss_remAMA = 2 * calc_remd_loss(real_B_feat[j],fake_B_feat[j])

        #####  color loss ######################

        color_loss = calc_histogram_loss(fake_B,real_B,hist)
        # ################  reconstruct line loss ##########################
        #
        # fake_A_line = (fake_A + 1) / 2
        # real_A_line = (real_A + 1) / 2
        #
        # fake_A_line = net_G_line(fake_A_line)
        # real_A_line = net_G_line(real_A_line)
        #
        # loss_rec_line = criterion_MSE(fake_A_line,real_A_line)

        # ################## patch style loss ##############################
        #
        # gray_real_B = transforms.functional.rgb_to_grayscale(real_B).repeat(1, 3, 1, 1)
        #
        # style_adaptive_alpha = (((cal_aec_loss.adaptive_gram_weight(real_B, 1, 8) + cal_aec_loss.adaptive_gram_weight(real_B, 2,8) +cal_aec_loss.adaptive_gram_weight(
        # real_B, 3, 8)) / 3).unsqueeze(1).cuda() + ((cal_aec_loss.adaptive_gram_weight(gray_real_B, 1, 8) + cal_aec_loss.adaptive_gram_weight(gray_real_B, 2, 8) + cal_aec_loss.adaptive_gram_weight(gray_real_B, 3, 8)) / 3).unsqueeze(1).cuda()) / 2
        #
        # loss_aes_style = cal_aec_loss.proposed_local_gram_loss_v2(fake_B,real_B,style_adaptive_alpha)

        ################ contruction patch loss #####################

        real_A_depth, fake_B_depth , loss_global_depth = cal_depth_object.cal_loss(fake_B,real_A_input)

        # real_A_depth, fake_B_depth , loss_global_depth = depth_gene.cal_depth_loss(real_A_input,fake_B)
        loss_global_depth = loss_global_depth * 5

        ################ ROPE loss #####################
        #
        # a_1 , a_2 = rope_code.Create_patch_feature(real_A_depth,14)
        # b_1 , b_2 = rope_code.Create_patch_feature(fake_B_depth,14)
        #
        # a_3 = rope_code(a_1,a_2)
        # b_3 = rope_code(b_1,b_2)
        #
        # loss_rope = criterion_MSE(a_3,b_3)

        # ################################ cut loss #####################################
        #
        # # gray_real_A = transforms.functional.rgb_to_grayscale(real_A).repeat(1, 3, 1, 1)
        # # depth_adaptive_alpha = (((patch_loss.adaptive_gram_weight(real_A, 1, 8)+patch_loss.adaptive_gram_weight(real_A, 2, 8)+patch_loss.adaptive_gram_weight(real_A_save, 3, 8) ) /3 ).unsqueeze(1).cuda() +\
		# # 	 ((patch_loss.adaptive_gram_weight(gray_real_A, 1, 8)+patch_loss.adaptive_gram_weight(gray_real_A, 2, 8)+patch_loss.adaptive_gram_weight(gray_real_A, 3, 8) ) /3 ).unsqueeze(1).cuda() )/2
        # # print(depth_adaptive_alpha)
        # # loss_cut = patch_loss.proposed_local_gram_loss_v2(real_A,fake_B,depth_adaptive_alpha)
        # # loss_cut = patch_loss.cal_nce_patchloss(real_A,fake_B) * 0.1
        # A_depth = real_A_depth.unsqueeze(1)
        # B_depth = fake_B_depth.unsqueeze(1)
        # if A_depth.size()[1] == 1:
        #     A_depth = A_depth.repeat(1, 3, 1, 1)
        # if B_depth.size()[1] == 1:
        #     B_depth = B_depth.repeat(1, 3, 1, 1)
        #
        # loss_cut = patch_loss.cal_nce_patchloss(A_depth, B_depth) * 0.1
        ########## loss A Reconstruction ##########

        loss_G_A = criterionGAN(netD_A(fake_A), True)

        # GAN loss D_B(G_B(B))
        loss_G_B = 0
        pred_fake_GAN = netD_B(fake_B)
        loss_G_B = criterionGAN(netD_B(fake_B), True)

        # Forward cycle loss || G_B(G_A(A)) - A||
        loss_cycle_A = criterionCycle(rec_A, real_A_input)
        loss_cycle_B = criterionCycleB(rec_B, real_B)
        # combined loss and calculate gradients

        loss_GAN = loss_G_A + loss_G_B
        loss_RC = loss_cycle_A + loss_cycle_B

        loss_G = cond_cycle*loss_RC + condGAN*loss_GAN
        # loss_G += opt.condGeom * loss_cycle_Geom

        loss_G = loss_G + loss_ada_local + loss_ada_gobal + loss_ada_content  +  color_loss  + loss_global_depth  # + loss_contentAMA + loss_remAMA

        ### semantic loss
        loss_recog = 0

        # renormalize mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)
        recog_real = real_A_input
        recog_real0 = (recog_real[:, 0, :, :].unsqueeze(1) - 0.48145466) / 0.26862954
        recog_real1 = (recog_real[:, 1, :, :].unsqueeze(1) - 0.4578275) / 0.26130258
        recog_real2 = (recog_real[:, 2, :, :].unsqueeze(1) - 0.40821073) / 0.27577711
        recog_real = torch.cat([recog_real0, recog_real1, recog_real2], dim=1)

        line_input = fake_B
        if opt.output_nc == 1:
            line_input_channel0 = (line_input - 0.48145466) / 0.26862954
            line_input_channel1 = (line_input - 0.4578275) / 0.26130258
            line_input_channel2 = (line_input - 0.40821073) / 0.27577711
            line_input = torch.cat([line_input_channel0, line_input_channel1, line_input_channel2], dim=1)

        patches_r = [torch.nn.functional.interpolate(recog_real, size=224)]  #The resize operation on tensor.
        patches_l = [torch.nn.functional.interpolate(line_input, size=224)]

        ## patch based clip loss
        if opt.N_patches > 1:
            patches_r2, patches_l2 = createNRandompatches(recog_real, line_input, opt.N_patches, opt.patch_size)
            patches_r += patches_r2
            patches_l += patches_l2

        loss_recog = 0
        for patchnum in range(len(patches_r)):

            real_patch = patches_r[patchnum]
            line_patch = patches_l[patchnum]

            feats_r = clip_model.encode_image(real_patch).detach()
            feats_line = clip_model.encode_image(line_patch)

            myloss_recog = criterionCLIP(feats_line, feats_r.detach())
            if opt.cos_clip == 1:
                myloss_recog = 1.0 - loss_recog
                myloss_recog = torch.mean(loss_recog)

            patch_factor = (1.0 / float(opt.N_patches))
            if patchnum == 0:
                patch_factor = 1.0
            loss_recog += patch_factor * myloss_recog
        loss_contrastive = 0






        if epoch >= 0 and opt.contrastive_weight > 0 :

            for t in range(opt.batchSize):
                fake_B_numpy = fake_B.clone()
                real_B_numpy = real_B.clone()
                to_pil = transforms.ToPILImage()
                fake_B_pil = to_pil(fake_B_numpy[t]).convert("L")
                real_B_pil = to_pil(real_B_numpy[t]).convert("L")

                fake_B_gray = np.array(fake_B_pil)
                real_B_gray = np.array(real_B_pil)

                fake_B_edges = cv2.Canny(fake_B_gray, 50, 100)
                real_B_edges = cv2.Canny(real_B_gray, 50, 100)
                fake_B_edges = Image.fromarray(fake_B_edges)
                real_B_edges = Image.fromarray(real_B_edges)
                totensor = transforms.ToTensor()
                fake_B_edges = totensor(fake_B_edges).unsqueeze(0)
                real_B_edges = totensor(real_B_edges).unsqueeze(0)
                dict_list  = get_tensor_patch(fake_B_edges,real_B_edges)
                len_patch = min(8,len(dict_list))
                for index in range(len_patch):
                    if dict_list == []:
                        break
                    a_x = dict_list[index]['a_x']
                    a_y = dict_list[index]['a_y']
                    b_x = dict_list[index]['b_x']
                    b_y = dict_list[index]['b_y']
                    fake_B_C = vggnet.encode_with_intermediate_level(fake_B[t,:,a_x:a_x+32,a_y:a_y+32].unsqueeze(0),2)
                    fake_B_C = gram_matrix(fake_B_C)
                    fake_B_C = vggnet.feature_normalize(fake_B_C)
                    fake_B_C = fake_B_C.view(fake_B_C.shape[0],-1)
                    real_B_C = vggnet.encode_with_intermediate_level(real_B[t,:,b_x:b_x+32,b_y:b_y+32].unsqueeze(0).detach(),2)
                    real_B_C = gram_matrix(real_B_C)
                    real_B_C = vggnet.feature_normalize(real_B_C)
                    real_B_C = real_B_C.view(real_B_C.shape[0],-1)
                    contrastive_loss = calc_contrastive_loss(fake_B_C, real_B_C, queue)
                    n = len(dict_list)
                    loss_contrastive += contrastive_loss * opt.contrastive_weight
                    loss_contrastive = loss_contrastive / n
                    fake_B_C = fake_B_C.detach()
                    queue = queue_data(queue, fake_B_C)
                    queue = dequeue_data(queue, K=1024)



        if epoch >= 0 and opt.contrastive_weight > 0 :
             loss_G += cond_recog * loss_recog  + loss_contrastive
        else:
             loss_G += cond_recog * loss_recog

        if opt.skip_connection_3:
            optim_atta_3.zero_grad()

        optim_atta_45.zero_grad()
        optimizer_G_A.zero_grad()
        optimizer_G_B.zero_grad()

        if opt.finetune_netGeom == 1:
            optimizer_Geom.zero_grad()


        loss_G.backward()
        if opt.skip_connection_3:
            optim_atta_3.step()
        optim_atta_45.step()
        optimizer_G_A.step()
        optimizer_G_B.step()
        if opt.finetune_netGeom == 1:
            optimizer_Geom.step()
        # depth_gene.update_model_state(loss_rope)
        ##########  Discriminator A ##########

        # Fake loss
        pred_fake_A = netD_A(fake_A.detach())
        loss_D_A_fake = criterionGAN(pred_fake_A, False)

        # Real loss

        pred_real_A = netD_A(real_A_input.detach())
        loss_D_A_real = criterionGAN(pred_real_A, True)

        # Total loss
        loss_D_A = torch.mean(condGAN * (loss_D_A_real + loss_D_A_fake) ) * 0.5

        optimizer_D_A.zero_grad()
        loss_D_A.backward()
        optimizer_D_A.step()

        ##########  Discriminator B ##########

        # Fake loss
        pred_fake_B = netD_B(fake_B.detach())
        loss_D_B_fake = criterionGAN(pred_fake_B, False)

        # Real loss

        pred_real_B = netD_B(real_B.detach())
        loss_D_B_real = criterionGAN(pred_real_B, True)

        # Total loss
        loss_D_B = torch.mean(condGAN * (loss_D_B_real + loss_D_B_fake) ) * 0.5

        optimizer_D_B.zero_grad()
        loss_D_B.backward()
        optimizer_D_B.step()

        real_A_depth = real_A_depth * 255.0
        real_A_depth_list = torch.split(real_A_depth,1,dim=0)
        real_A_save = torch.cat(real_A_depth_list,dim=2)
        real_A_save = real_A_save.cpu().numpy().astype(np.uint8)
        real_A_save = (cv2.applyColorMap( real_A_save[0], cv2.COLORMAP_INFERNO))
        # real_A_save = (cmap(real_A_save[0]), cv2.COLORMAP_INFERNO)
        # print(real_A_save)
        # print(real_A_save.shape)
        cv2.imwrite('/home/yifan/yf/informative-finally3/val/' + 'real_A_depth_epoch%02d.jpg' % (epoch),real_A_save)



        fake_B_depth = fake_B_depth * 255.0   # b,c,h,w  b,1,266,266
        fake_B_depth_list = torch.split(fake_B_depth, 1, dim=0)     # (1,266,266)
        fake_B_save = torch.cat(fake_B_depth_list , dim=2)
        fake_B_save = fake_B_save.cpu().numpy().astype(np.uint8)
        # fake_B_save = (cmap(fake_B_save[0]), cv2.COLORMAP_INFERNO)
        fake_B_save = (cv2.applyColorMap(fake_B_save[0], cv2.COLORMAP_INFERNO))
        cv2.imwrite('/home/yifan/yf/informative-finally3/val/' + 'fake_B_depth_epoch%02d.jpg' % (epoch),fake_B_save)


        rec_A_list = torch.split(rec_A, 1, dim=0)
        rec_A_save = torch.cat(rec_A_list, dim=3)
        save_image(rec_A_save, '/home/yifan/yf/informative-finally3/val/' + 'recA_epoch%02d.jpg' % (epoch))

        rec_B_list = torch.split(rec_B, 1, dim=0)
        rec_B_save = torch.cat(rec_B_list, dim=3)
        save_image(rec_B_save, '/home/yifan/yf/informative-finally3/val/' + 'recB_epoch%02d.jpg' % (epoch))


        fake_A_list = torch.split(fake_A, 1, dim=0)
        fake_A_save = torch.cat(fake_A_list, dim=3)
        save_image(fake_A_save, '/home/yifan/yf/informative-finally3/val/' + 'fakeA_epoch%02d.jpg' % (epoch))

        real_A_list = torch.split(real_A, 1, dim=0)
        real_A_save = torch.cat(real_A_list, dim=3)
        save_image(real_A_save, '/home/yifan/yf/informative-finally3/val/' + 'realA_epoch%02d.jpg' % (epoch))


        fake_B_list = torch.split(fake_B, 1, dim=0)
        fake_B_save = torch.cat(fake_B_list, dim=3)
        save_image(fake_B_save, '/home/yifan/yf/informative-finally3/val/' + 'fakeB_epoch%02d.jpg' % (epoch))


        real_B_list = torch.split(real_B, 1, dim=0)
        real_B_save = torch.cat(real_B_list, dim=3)
        save_image(real_B_save, '/home/yifan/yf/informative-finally3/val/' + 'realB_epoch%02d.jpg' % (epoch))

        if (i + 1) % opt.log_int == 0:
            # 打印损失函数
            errors = {}
            errors['total_G'] = loss_G.item() if not isinstance(loss_G, (int, float)) else loss_G
            # errors['loss_aesstyle'] = loss_aes_style.item() if not isinstance(loss_aes_style, (int, float)) else loss_aes_style
            errors['loss_RC'] = torch.mean(loss_RC) if not isinstance(loss_RC, (int, float)) else loss_RC
            # errors['loss_cut'] = torch.mean(loss_cut) if not isinstance(loss_cut, (int,float)) else loss_cut
            errors['loss_depth'] = torch.mean(loss_global_depth) if not isinstance(loss_global_depth,
                                                                                   (int, float)) else loss_global_depth
            errors['loss_GAN'] = torch.mean(loss_GAN) if not isinstance(loss_GAN, (int, float)) else loss_GAN
            errors['loss_D_B'] = loss_D_B.item() if not isinstance(loss_D_B, (int, float)) else loss_D_B
            errors['loss_D_A'] = loss_D_A.item() if not isinstance(loss_D_A, (int, float)) else loss_D_A
            errors['loss_ada_local'] = loss_ada_local.item() if not isinstance(loss_ada_local,
                                                                               (int, float)) else loss_ada_local
            errors['loss_ada_gobal'] = loss_ada_gobal.item() if not isinstance(loss_ada_gobal,
                                                                               (int, float)) else loss_ada_gobal
            errors['color_loss'] = color_loss.item() if not isinstance(color_loss,
                                                                               (int, float)) else color_loss
            errors['loss_ada_content'] = loss_ada_content.item() if not isinstance(loss_ada_content,
                                                                                   (int, float)) else loss_ada_content
            # errors['loss_contentAMA'] = loss_contentAMA.item() if not isinstance(loss_contentAMA,
            #                                                                        (int, float)) else loss_contentAMA
            #
            # errors['loss_remAMA'] =loss_remAMA.item() if not isinstance(loss_remAMA,
            #                                                            (int, float)) else loss_remAMA
            # errors['loss_rec_line'] = loss_rec_line.item() if not isinstance(loss_rec_line, (int, float)) else loss_rec_line
            # errors['loss_rope'] = loss_rope.item() if not isinstance(loss_rope,(int, float)) else loss_rope
            if epoch >= 0 and opt.contrastive_weight > 0:
                errors['loss_contrastive'] = loss_contrastive if not isinstance(loss_contrastive,
                                                                                (int, float)) else loss_contrastive
            visualizer.print_current_errors(epoch, total_steps, errors, 0)
            visualizer.plot_current_errors(errors, total_steps)

            with torch.no_grad():

                input_img = channel2width(real_A)
                if opt.use_geom == 1:
                    pred_geom = channel2width(pred_geom)
                    input_img = torch.cat([input_img, channel2width(recover_geom)], dim=3)

                input_img_fake = channel2width(fake_A)
                rec_A = channel2width(rec_A)

                show_real_B = real_B

                visuals = OrderedDict([('real_A', tensor2im(input_img.data[0])),
                                           ('real_B', tensor2im(show_real_B.data[0])),
                                           ('fake_A', tensor2im(input_img_fake.data[0])),
                                           ('rec_A', tensor2im(rec_A.data[0])),
                                           ('fake_B', tensor2im(fake_B.data[0]))])

                if opt.use_geom == 1:
                    visuals['pred_geom'] = tensor2im(pred_geom.data[0])

                visualizer.display_current_results(visuals, total_steps, epoch)


    # Update learning rates
    lr_scheduler_G_A.step()
    lr_scheduler_G_B.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()
    if opt.skip_connection_3:
        lr_scheduler_attn3.step()

    lr_scheduler_attn45.step()
    # Save models checkpoints
    # torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')
    if (epoch+1) % opt.save_epoch_freq == 0:
        torch.save(netG_A.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netG_A_%02d.pth'%(epoch)))
        if opt.skip_connection_3:
            torch.save(net_adaattn_3.state_dict(), os.path.join(opt.checkpoints_dir, name, 'adaattn3_%02d.pth' % (epoch)))
        torch.save(net_transformer.state_dict(), os.path.join(opt.checkpoints_dir, name, 'transformer_%02d.pth' % (epoch)))
        if opt.finetune_netGeom == 1:
            torch.save(netGeom.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netGeom_%02d.pth'%(epoch)))
        if opt.slow == 0:
            torch.save(netG_B.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netG_B_%02d.pth'%(epoch)))
            torch.save(netD_A.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netD_A_%02d.pth'%(epoch)))
            torch.save(netD_B.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netD_B_%02d.pth'%(epoch)))
    # depth_gene.save_model(epoch=epoch)
    torch.save(netG_A.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netG_A_latest.pth'))
    torch.save(netG_B.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netG_B_latest.pth'))
    torch.save(netD_B.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netD_B_latest.pth'))
    torch.save(netD_A.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netD_A_latest.pth'))
    if opt.skip_connection_3:
        torch.save(net_adaattn_3.state_dict(), os.path.join(opt.checkpoints_dir, name, 'adaattn3_latest.pth'))
    torch.save(net_transformer.state_dict(), os.path.join(opt.checkpoints_dir, name, 'transformer_latest.pth'))
    if opt.finetune_netGeom == 1:
        torch.save(netGeom.state_dict(), os.path.join(opt.checkpoints_dir, name, 'netGeom_latest.pth'))

###################################
