
import torch
import torch.nn as nn
import math

class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class PatchSampleF(nn.Module):
    def __init__(self):
        # potential issues: currently, we use the same patch_ids for multiple images in the batch
        super(PatchSampleF, self).__init__()
        self.l2norm = Normalize(2)

    def forward(self, feat, num_patches=64, patch_ids=None):
        feat_reshape = feat.permute(0, 2, 3, 1).flatten(1, 2)
        if patch_ids is not None:
            patch_id = patch_ids
        else:
            patch_id = torch.randperm(feat_reshape.shape[1], device=feat[0].device)
            patch_id = patch_id[:int(min(num_patches, patch_id.shape[0]))]  # .to(patch_ids.device)
        x_sample = feat_reshape[:, patch_id, :].flatten(0, 1)  # reshape(-1, x.shape[1])
        x_sample = self.l2norm(x_sample)
        # return_feats.append(x_sample)
        return x_sample, patch_id

class BidirectionalNCE1(nn.Module):
    def __init__(self):
        super().__init__()
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.bool #torch.uint8 if version.parse(torch.__version__)
        self.cosm = math.cos(0.25)
        self.sinm = math.sin(0.25)

    def forward(self, feat_q, feat_k):

        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1))
        l_pos = l_pos.view(batchSize, 1)

        cosx = l_pos
        sinx = torch.sqrt(1.0 - torch.pow(cosx, 2))
        l_pos = cosx * self.cosm - sinx * self.sinm

        batch_dim_for_bmm = int(batchSize / 64)
        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        # just fill the diagonal with very small number, which is exp(-10)
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / 0.05

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device)).mean()

        return loss
if __name__ == '__main__':
    nceLoss = BidirectionalNCE1()
    patch_sample = PatchSampleF()

    feature_gt = torch.rand((4,512,16,16))
    feature_output = torch.rand((4, 512, 16, 16))
    feat_k, sample_ids = patch_sample(feature_output, 64, None)
    feat_q, _ = patch_sample(feature_gt, 64, sample_ids)

    nceloss = nceLoss(feat_k, feat_q)
    print(nceloss)
