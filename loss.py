import torch
import torch.nn as nn
import torch.nn.functional as F
import  random
import  Vggnet
from LOSSNEC  import *
class Calc_Style_Loss:

    def calc_style_loss(self, input, target):
        assert (input.size() == target.size())
        assert (target.requires_grad is False)
        self.mse_loss = nn.MSELoss()
        input_mean, input_std = self.calc_mean_std(input)
        target_mean, target_std = self.calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + \
               self.mse_loss(input_std, target_std)

    def calc_mean_std(self,feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
        return feat_mean, feat_std

    def final_loss(self,output,refer):
        loss_s = self.calc_style_loss(output[0], refer[0])
        for i in range(1, 4):
            loss_s += self.calc_style_loss(output[i], refer[i])
        return loss_s

def gram_matrix(y):

	(b, ch, h, w) = y.size()
	features = y.view(b, ch, w * h)
	features_t = features.transpose(1, 2)
	gram = features.bmm(features_t) / (ch * h * w)
	return gram

class Patch_loss:
    def __init__(self):
        self.vggnet = Vggnet.Vgg_net()

    def extract_image_patches(self, x, kernel, stride=1):
        b, c, h, w = x.shape

        # Extract patches
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.contiguous().view(b, c, -1, kernel, kernel)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()

        return patches.view(b, -1, c, kernel, kernel)

    # return patches.view(b, number_of_patches, c, h, w)

    def adaptive_gram_weight(self, image, level, ratio):
        if level == 0:
            encoded_features = image
        else:
            encoded_features = self.network.encoder.get_features(image, level)  # B x C x W x H
        global_gram = gram_matrix(encoded_features)

        B, C, w, h = encoded_features.size()
        target_w, target_h = w // ratio, h // ratio
        # assert target_w==target_h

        patches = self.extract_image_patches(encoded_features, target_w, target_h)
        _, patches_num, _, _, _ = patches.size()
        cos = torch.nn.CosineSimilarity(eps=1e-6)

        intra_gram_statistic = []
        inter_gram_statistic = []
        comb = torch.combinations(torch.arange(patches_num), r=2)
        if patches_num >= 10:
            sampling_num = int(comb.size(0) * 0.05)
        else:
            sampling_num = comb.size(0)
        for idx in range(B):
            if patches_num < 2:
                continue
            cos_gram = []

            for patch in range(0, patches_num):
                cos_gram.append(cos(global_gram, gram_matrix(patches[idx][patch].unsqueeze(0))).mean().item())

            intra_gram_statistic.append(torch.tensor(cos_gram))

            cos_gram = []
            for idxes in random.choices(list(comb), k=sampling_num):
                cos_gram.append(cos(gram_matrix(patches[idx][idxes[0]].unsqueeze(0)),
                                    gram_matrix(patches[idx][idxes[1]].unsqueeze(0))).mean().item())

            inter_gram_statistic.append(torch.tensor(cos_gram))

        intra_gram_statistic = torch.stack(intra_gram_statistic).mean(dim=1)
        inter_gram_statistic = torch.stack(inter_gram_statistic).mean(dim=1)
        results = (intra_gram_statistic + inter_gram_statistic) / 2

        ##For boosting value
        results = (1 / (1 + torch.exp(-10 * (results - 0.6))))

        return results

    def calc_contrastive_loss(self, query, key, queue, temp=0.07):
        N = query.shape[0]
        K = queue.shape[0]

        zeros = torch.zeros(N, dtype=torch.long, device=query.device)
        key = key.detach()

        logit_pos = torch.bmm(query.view(N, 1, -1), key.view(N, -1, 1))
        logit_neg = torch.mm(query.view(N, -1), queue.t().view(-1, K))

        logit = torch.cat([logit_pos.view(N, 1), logit_neg], dim=1)

        loss = F.cross_entropy(logit / temp, zeros)

        return loss

    def proposed_local_gram_loss_v2(self, content, output, alpha):
        local_content_loss = 0
        B, C, th, tw = output.size()
        # window_size = min(int(2 ** int((9 / 8 - alpha[B]) * 8 + 4)), 256)
        # for batch in range(B):
        #     window_size = min(int(2 ** int((9 / 8 - alpha[batch]) * 8 + 4)), 256)
        #     for level in [4, 5]:
        #         content_patches = self.vggnet.encode_with_intermediate_level(
        #             self.extract_image_patches(content[batch:batch + 1], window_size, window_size).squeeze(0),
        #             level)
        #
        #             output_patches = self.vggnet.encode_with_intermediate_level(
        #                 self.extract_image_patches(content[batch:batch + 1], window_size, window_size).squeeze(0), level)
        # content_patches = self.extract_image_patches(content,window_size,window_size)   #  (b, patch_nums, c, kernel, kernel)
        # output_patches = self.extract_image_patches(output,window_size,window_size)    #  (b, patch_nums , c, kernel, kernel)
        # for level in [4,5]:
        #     pos = []
        #     neg = []
        #     patch_num = content_patches[1]
        #     for i in range(patch_num):
        #             content_patches_feature = self.vggnet.encode_with_intermediate_level(content_patches[:,i:i+1,:,:,:].squeeze(1))   # b, N ,h ,w
        #             for j in range(patch_num):
        #                  output_patches_feature = self.vggnet.encode_with_intermediate_level(output_patches[:,j:j+1,:,:,:].squeeze(1))  # b, N ,h ,w
        #                  if i == j:
        #                     pos.append(output_patches_feature)
        #                 neg.append(content_patches_feature)


        return

    def cal_nce_patchloss(self,content,output):

        nceLoss = BidirectionalNCE1()
        patch_sample = PatchSampleF()
        level_nceloss = 0
        for i in [4,5]:
            feature_gt = self.vggnet.encode_with_intermediate_level(content,i)
            feature_output = self.vggnet.encode_with_intermediate_level(output,i)
            feat_k, sample_ids = patch_sample(feature_output, 64, None)
            feat_q, _ = patch_sample(feature_gt, 64, sample_ids)

            level_nceloss += nceLoss(feat_k, feat_q)
        level_nceloss = level_nceloss / 2

        return  level_nceloss