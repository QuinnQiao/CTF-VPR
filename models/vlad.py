import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


__all__ = ['NetVLAD', 'EmbedRegionNet']


class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=512, normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.normalize_input = normalize_input

        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=False)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim), requires_grad=True)

    def forward(self, x):
        N, C = x.shape[:2]
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1) # NCHW
        
        # soft-assignment, NKHW -> NKL (where L=HW)
        soft_assign = F.softmax(self.conv(x).view(N, self.num_clusters, -1), dim=1)
        # NCHW -> NCL 
        x_flatten = x.view(N, C, -1)
        # final vlad vector, NKC
        vlad = torch.zeros([N, self.num_clusters, C], dtype=x.dtype, device=x.device)
        for i in range(self.num_clusters):
            # (NCL - 1C1) * N1L
            residual = (x_flatten - self.centroids[i].view(1, -1, 1)) * soft_assign[:, i, :].view(N, 1, -1)
            # NCL -> NC
            vlad[:, i, :] = residual.sum(dim=-1)

        return vlad
    

class EmbedRegionNet(nn.Module):
    def __init__(self, base_model, net_vlad, subratio=0.65):
        super(EmbedRegionNet, self).__init__()
        self.base_model = base_model
        self.net_vlad = net_vlad
        if subratio < 0.5 or subratio >= 1:
            raise ValueError('Illegal sub-region ratio: {}'.format(subratio))
        self.subratio = subratio

    def reset_ratio(self, subratio):
        self.subratio = subratio

    def _compute_overlap_region_vlad(self, x, keep_aspect_ratio=False):
        subratio = np.sqrt(self.subratio) if keep_aspect_ratio else self.subratio
        
        _, _, H, W = x.size()
        h = int(H * subratio)
        _h = H - h
        w = int(W * subratio)
        _w = W - w
        # divide the full map into 9 sub-regions
        regions = [
            x[:, :, :_h,  :_w ],   # top left
            x[:, :, :_h,  _w:w],   # top center
            x[:, :, :_h,  w:  ],   # top right
            x[:, :, _h:h, :_w ],   # center left
            x[:, :, _h:h, _w:w],   # center center
            x[:, :, _h:h, w:  ],   # center right
            x[:, :, h:,   :_w ],   # bottom left
            x[:, :, h:,   _w:w],   # bottom center
            x[:, :, h:,   w:  ]    # bottom right
        ]
        vlads = [self.net_vlad(r) if (r.size(2) > 0 and r.size(3) > 0) else None for r in regions]
        
        def sum_with_none(indice):
            return sum(vlads[i] for i in indice if vlads[i] is not None)

        # merge subregions into final regions
        if keep_aspect_ratio:
            regions_vlad = [
                sum_with_none([0, 1, 3, 4]),
                sum_with_none([1, 2, 4, 5]),
                sum_with_none([3, 4, 6, 7]),
                sum_with_none([4, 5, 7, 8]),
                sum_with_none([0, 1, 2, 3, 4, 5, 6, 7, 8]) # append the full map
            ]
        elif H <= W:
            regions_vlad = [
                sum_with_none([0, 1, 3, 4, 6, 7]),
                sum_with_none([1, 2, 4, 5, 7, 8])
            ]
        else:
            regions_vlad = [
                sum_with_none([0, 1, 2, 3, 4, 5]),
                sum_with_none([3, 4, 5, 6, 7, 8])
            ]

        return regions_vlad
        
    def forward(self, x):
        x = self.base_model(x)
        vlad_x = self._compute_overlap_region_vlad(x, keep_aspect_ratio=False) # without the full map
        vlad_x.extend(self._compute_overlap_region_vlad(x, keep_aspect_ratio=True)) # x[-1] if the full map

        vlad_x = torch.stack(vlad_x, dim=1) # N9CK
        N, B, _, _ = vlad_x.size()
        
        vlad_x = F.normalize(vlad_x, p=2, dim=3)
        vlad_x = vlad_x.view(N, B, -1)
        vlad_x = F.normalize(vlad_x, p=2, dim=2)
        
        return vlad_x # N9L
