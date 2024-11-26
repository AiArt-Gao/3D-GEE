import random
import torch
import Vggnet
class cal_style_patch:
    def __init__(self):
        self.vgg = Vggnet.Vgg_net()

        self.MSE_loss = torch.nn.MSELoss(reduce=True)
    def gram_matrix(self,y):
        (b, ch, h, w) = y.size()
        features = y.view(b, ch, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (ch * h * w)
        return gram


    def adaptive_gram_weight(self, image, level, ratio):
        if level == 0:
            encoded_features = image
        else:
            encoded_features =self.vgg.encode_with_intermediate_level(image, level)  # B x C x W x H
        global_gram = self.gram_matrix(encoded_features)

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
                cos_gram.append(cos(global_gram, self.gram_matrix(patches[idx][patch].unsqueeze(0))).mean().item())

            intra_gram_statistic.append(torch.tensor(cos_gram))

            cos_gram = []
            for idxes in random.choices(list(comb), k=sampling_num):
                cos_gram.append(cos(self.gram_matrix(patches[idx][idxes[0]].unsqueeze(0)),
                                    self.gram_matrix(patches[idx][idxes[1]].unsqueeze(0))).mean().item())

            inter_gram_statistic.append(torch.tensor(cos_gram))

        intra_gram_statistic = torch.stack(intra_gram_statistic).mean(dim=1)
        inter_gram_statistic = torch.stack(inter_gram_statistic).mean(dim=1)
        results = (intra_gram_statistic + inter_gram_statistic) / 2

        ##For boosting value
        results = (1 / (1 + torch.exp(-10 * (results - 0.6))))

        return results


    def extract_image_patches(self, x, kernel, stride=1):
        b, c, h, w = x.shape

        # Extract patches
        patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)
        patches = patches.contiguous().view(b, c, -1, kernel, kernel)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()

        return patches.view(b, -1, c, kernel, kernel)


    # return patches.view(b, number_of_patches, c, h, w)
    def proposed_local_gram_loss_v2(self, output, style, alpha):
        local_style_loss = 0

        B, C, th, tw = style.size()
        for batch in range(B):
            window_size = min(int(2 ** int((9 / 8 - alpha[batch]) * 8 + 4)), 256)
            # window_size = alpha
            for level in [4, 5]:
                output_patches = self.vgg.encode_with_intermediate_level(
                    self.extract_image_patches(output[batch:batch + 1], window_size, window_size).squeeze(0), level)
                style_patches = self.vgg.encode_with_intermediate_level(
                    self.extract_image_patches(style[batch:batch + 1], window_size, window_size).squeeze(0), level)

                gram_stylization_patches = self.gram_matrix(output_patches - torch.mean(output_patches))
                gram_style_patches = self.gram_matrix(style_patches - torch.mean(style_patches))

                local_style_loss += self.MSE_loss(gram_stylization_patches, gram_style_patches)

        return local_style_loss / B / 2

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = torch.bmm(features_t,features) / (ch * h * w)
    return gram

if __name__ == '__main__':

    CUDA_VISIBLE_DEVICES = '1'
    a = torch.rand((3,3,16,16)).to(1)
    d  = gram_matrix(a)


    print(d.shape)