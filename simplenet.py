import torch
import torch.nn as nn

import resnet


class Projection(nn.Module):
    def __init__(self,
                 dim=1600,
                 ):
        super(Projection, self).__init__()
        self.dim = dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        return self.proj(x)


class Discriminator(nn.Module):
    def __init__(self,
                 input_dim=1600,
                 ):
        super(Discriminator, self).__init__()
        self.input_dim = input_dim

        self.model = nn.Sequential(
            nn.Linear(input_dim, 1200),
            nn.BatchNorm1d(1200),
            nn.LeakyReLU(0.2),
            nn.Linear(1200, 1, bias=False)
        )

    def forward(self, x):
        return self.model(x)



class SimpleNet(nn.Module):
    def __init__(self,
                 channels=3,
                 img_size=224,
                 embed_dim=1600
                 ):
        super(SimpleNet, self).__init__()

        self.img_size = img_size
        self.layers_to_extract_from = ['layer2', 'layer3']
        self.embed_dim = embed_dim

        self.extractor = resnet.wide_resnet50_2(True)
        self.extractor.eval()
        self.projection = Projection(self.embed_dim)
        self.discriminator = Discriminator(self.embed_dim)

        self.patch_maker = PatchMaker(self.img_size)

    def embed(self, img):
        with torch.no_grad():
            features = self.extractor(img)
        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]



class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=1):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """Convert a tensor into a tensor of respective patches.
        Args:
            x: [torch.Tensor, bs x c x w x h]
        Returns:
            x: [torch.Tensor, bs * w//stride * h//stride, c, patchsize,
            patchsize]
        """
        padding = int((self.patchsize - 1) / 2)
        unfolder = torch.nn.Unfold(
            kernel_size=self.patchsize, stride=self.stride, padding=padding, dilation=1
        )
        unfolded_features = unfolder(features)
        number_of_total_patches = []
        for s in features.shape[-2:]:
            n_patches = (
                s + 2 * padding - 1 * (self.patchsize - 1) - 1
            ) / self.stride + 1
            number_of_total_patches.append(int(n_patches))
        unfolded_features = unfolded_features.reshape(
            *features.shape[:2], self.patchsize, self.patchsize, -1
        )
        unfolded_features = unfolded_features.permute(0, 4, 1, 2, 3)

        if return_spatial_info:
            return unfolded_features, number_of_total_patches
        return unfolded_features

    def unpatch_scores(self, x, batchsize):
        return x.reshape(batchsize, -1, *x.shape[1:])
