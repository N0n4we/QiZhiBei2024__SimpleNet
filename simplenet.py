import torch
import torch.nn as nn
import torch.nn.functional as F
import resnet
import common

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
                 embed_dim=1600,
                 input_shape=(3, 224, 224),
                 device='gpu',
                 ):
        super(SimpleNet, self).__init__()

        self.img_size = img_size
        self.layers_to_extract_from = ['layer2', 'layer3']
        self.embed_dim = embed_dim
        self.device = device = torch.device(device)

        self.backbone = resnet.wide_resnet50_2(True).to(device)
        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone=False
        )
        self.forward_modules["feature_aggregator"] = feature_aggregator
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape=input_shape)
        preprocessing = common.Preprocessing(
            feature_dimensions, embed_dim
        )
        self.forward_modules["preprocessing"] = preprocessing
        preadapt_aggregator = common.Aggregator(
            target_dim=embed_dim
        )
        _ = preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator
        self.projection = Projection(self.embed_dim).to(device)
        self.discriminator = Discriminator(self.embed_dim).to(device)

        self.patch_maker = PatchMaker(self.img_size)
        self.anomaly_segmentor = common.RescaleSegmentor(
            device=self.device, target_size=input_shape[-2:]
        )

    def embed(self, img):
        with torch.no_grad():
            features = self.extractor(img)
        features = [features[layer] for layer in self.layers_to_extract_from]

        features = [
            self.patch_maker.patchify(x, return_spatial_info=True) for x in features
        ]
        patch_shapes = [x[1] for x in features]
        features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(features)):
            _features = features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            features[i] = _features
        features = [x.reshape(-1, *x.shape[-3:]) for x in features]

        # As different feature backbones & patching provide differently
        # sized features, these are brought into the correct form here.
        features = self.forward_modules["preprocessing"](
            features)  # pooling each feature to same channel and stack together
        features = self.forward_modules["preadapt_aggregator"](features)  # further pooling

        return features, patch_shapes
    def forward(self, images, mode='eval'):
        """Infer score and mask for a batch of images."""
        images = images.to(torch.float).to(self.device)
        _ = self.forward_modules.eval()

        batchsize = images.shape[0]
        if mode == 'eval':
            self.pre_projection.eval()
            self.discriminator.eval()
            with torch.no_grad():
                features, patch_shapes = self.embed(images)
                features = self.pre_projection(features)
                scores = self.discriminator(features)
        elif mode == 'train':
            self.pre_projection.train()
            self.discriminator.train()
            features, patch_shapes = self.embed(images)
            features = self.pre_projection(features)
            '''
            TODO: Add Noise
            '''
            scores = self.discriminator(features)
        return scores


class PatchMaker:
    def __init__(self, patchsize, top_k=0, stride=1):
        self.patchsize = patchsize
        self.stride = stride
        self.top_k = top_k

    def patchify(self, features, return_spatial_info=False):
        """
        Convert a tensor into a tensor of respective patches.
        Args:
            features: [torch.Tensor, bs x c x w x h]
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

    def score(self, x):
        x = torch.from_numpy(x)
        while x.ndim > 2:
            x = torch.max(x, dim=-1).values
        if x.ndim == 2:
            if self.top_k > 1:
                x = torch.topk(x, self.top_k, dim=1).values.mean(1)
            else:
                x = torch.max(x, dim=1).values
        return x.numpy()
