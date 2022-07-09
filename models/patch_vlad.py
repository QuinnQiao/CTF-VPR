import torch
from torch import nn
from torch.nn import functional as F


__all__ = ['PatchNetVLAD', 'EmbedPatchNet']


def get_integral_feature(feat_in):
    """
    Input/Output as [N,D,H,W] where N is batch size and D is descriptor dimensions
    For VLAD, D = K x d where K is the number of clusters and d is the original descriptor dimensions

    """
    feat_out = torch.cumsum(feat_in, dim=-1)
    feat_out = torch.cumsum(feat_out, dim=-2)
    feat_out = torch.nn.functional.pad(feat_out, (1, 0, 1, 0), "constant", 0)
    return feat_out


def get_square_regions_from_integral(feat_integral, patch_size, patch_stride):
    """
    Input as [N,D,H+1,W+1] where additional 1s for last two axes are zero paddings
    regSize and regStride are single values as only square regions are implemented currently

    """
    N, D, H, W = feat_integral.shape

    if feat_integral.get_device() == -1:
        conv_weight = torch.ones(D, 1, 2, 2)
    else:
        conv_weight = torch.ones(D, 1, 2, 2, device=feat_integral.get_device())
    conv_weight[:, :, 0, -1] = -1
    conv_weight[:, :, -1, 0] = -1
    feat_regions = torch.nn.functional.conv2d(feat_integral, conv_weight, stride=patch_stride, groups=D, dilation=patch_size)
    return feat_regions / (patch_size ** 2)


class PatchNetVLAD(nn.Module):
    def __init__(self, num_clusters=64, dim=128, normalize_input=True, patch_size=5, stride=1):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
            patch_sizes: string
                comma separated string of patch sizes
            strides: string
                comma separated string of strides (for patch aggregation)
        """
        super().__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        
        self.patch_size = patch_size
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = F.softmax(self.conv(x), dim=1) # NKHW

        # calculate residuals to each cluster
        store_residual = torch.zeros([N, self.num_clusters, C, H, W], dtype=x.dtype, device=x.device)
        for j in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            # (NCHW - 1C11) * N1HW => NCHW
            residual = (x - self.centroids[j].view(1, -1, 1, 1)) * soft_assign[:, j, :, :].unsqueeze(1)
            store_residual[:, j, :, :, :] = residual

        store_residual = store_residual.view(N, -1, H, W) # N(KC)HW
        store_residual = get_integral_feature(store_residual) # N(KC)HW

        store_residual = get_square_regions_from_integral(store_residual, self.patch_size, self.stride) # N(KC)HW

        return store_residual


class EmbedPatchNet(nn.Module):
    def __init__(self, base_model, net_vlad):
        super(EmbedPatchNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        self.feature_dim = self.base_model.feature_dim * self.net_vlad.num_clusters

    def forward(self, x):
        x = self.base_model(x) # NCHW
        vlad_x = self.net_vlad(x) # N(KC)HW

        # [IMPORTANT] normalize
        vlad_x = vlad_x.view(vlad_x.size(0), self.net_vlad.num_clusters, self.base_model.feature_dim, -1) # NKCL
        vlad_x = F.normalize(vlad_x, p=2, dim=2)
        vlad_x = vlad_x.view(vlad_x.size(0), -1, vlad_x.size(3)) # N(KC)L
        vlad_x = F.normalize(vlad_x, p=2, dim=1)
        
        return vlad_x
