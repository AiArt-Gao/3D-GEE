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


class Ada_way():
    def __init__(self,opt):
        self.opt = opt
        self.cs = None
        self.s_feats = None
        self.c_feats = None
        self.seed = 6666
        self.device = "cuda"
        self.max_sample = 64 * 64
        self.criterionMSE = torch.nn.MSELoss().to(self.device)
        self.loss_global = torch.tensor(0., device=self.device)
        self.loss_local = torch.tensor(0., device=self.device)
        self.loss_content = torch.tensor(0., device=self.device)

    def feature_normalize(self, feature_in, eps=1e-10):
        feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + eps
        feature_in_norm = torch.div(feature_in, feature_in_norm)
        return feature_in_norm

    def encode_with_intermediate(self, input_img):
        results = [input_img]
        for i in range(5):
            func = self.image_encoder_layers[i]
            results.append(func(results[-1]))
        return results[1:]

    def get_key(self, feats, last_layer_idx, need_shallow=True):
        if need_shallow and last_layer_idx > 0:
            results = []
            _, _, h, w = feats[last_layer_idx].shape
            for i in range(last_layer_idx):
                results.append(adaattn.mean_variance_norm(nn.functional.interpolate(feats[i], (h, w))))
            results.append(adaattn.mean_variance_norm(feats[last_layer_idx]))
            return torch.cat(results, dim=1)
        else:
            return adaattn.mean_variance_norm(feats[last_layer_idx])

    def compute_content_loss(self, stylized_feats):
        self.loss_content = torch.tensor(0., device=self.device)
        if self.opt.lambda_content > 0:
            for i in range(1, 5):
                self.loss_content += self.criterionMSE(adaattn.mean_variance_norm(stylized_feats[i]),
                                                       adaattn.mean_variance_norm(self.c_feats[i]))

    def compute_losses(self):
        stylized_feats = self.cs
        self.compute_content_loss(stylized_feats)
        self.compute_style_loss(stylized_feats)
        self.loss_content = self.loss_content * self.opt.lambda_content
        self.loss_local = self.loss_local * self.opt.lambda_local
        self.loss_global = self.loss_global * self.opt.lambda_global

    def compute_style_loss(self, stylized_feats):
        self.loss_global = torch.tensor(0., device=self.device)
        if self.opt.lambda_global > 0:
            for i in range(1, 5):
                s_feats_mean, s_feats_std = adaattn.calc_mean_std(self.s_feats[i])
                stylized_feats_mean, stylized_feats_std = adaattn.calc_mean_std(stylized_feats[i])
                self.loss_global += self.criterionMSE(
                    stylized_feats_mean, s_feats_mean) + self.criterionMSE(stylized_feats_std, s_feats_std)
        self.loss_local = torch.tensor(0., device=self.device)
        if self.opt.lambda_local > 0:
            for i in range(1, 5):
                c_key = self.get_key(self.c_feats, i, self.opt.shallow_layer)
                s_key = self.get_key(self.s_feats, i, self.opt.shallow_layer)
                s_value = self.s_feats[i]
                b, _, h_s, w_s = s_key.size()
                s_key = s_key.view(b, -1, h_s * w_s).contiguous()
                if h_s * w_s > self.max_sample:
                    torch.manual_seed(self.seed)
                    index = torch.randperm(h_s * w_s).to(self.device)[:self.max_sample]
                    s_key = s_key[:, :, index]
                    style_flat = s_value.view(b, -1, h_s * w_s)[:, :, index].transpose(1, 2).contiguous()
                else:
                    style_flat = s_value.view(b, -1, h_s * w_s).transpose(1, 2).contiguous()
                b, _, h_c, w_c = c_key.size()
                c_key = c_key.view(b, -1, h_c * w_c).permute(0, 2, 1).contiguous()
                attn = torch.bmm(c_key, s_key)
                # S: b, n_c, n_s
                attn = torch.softmax(attn, dim=-1)
                # mean: b, n_c, c
                mean = torch.bmm(attn, style_flat)
                # std: b, n_c, c
                std = torch.sqrt(torch.relu(torch.bmm(attn, style_flat ** 2) - mean ** 2))
                # mean, std: b, c, h, w
                mean = mean.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                std = std.view(b, h_c, w_c, -1).permute(0, 3, 1, 2).contiguous()
                self.loss_local += self.criterionMSE(stylized_feats[i],
                                                     std * adaattn.mean_variance_norm(self.c_feats[i]) + mean)

