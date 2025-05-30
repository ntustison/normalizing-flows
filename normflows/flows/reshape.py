import torch
import torch.nn as nn
import numpy as np

from .base import Flow


# Flow layers to reshape the latent features

class Split(Flow):
    """
    Split features into two sets
    """

    def __init__(self, mode="channel"):
        super().__init__()
        self.mode = mode

    def forward(self, z):
        if self.mode == "channel":
            z1, z2 = z.chunk(2, dim=1)
        elif self.mode == "channel_inv":
            z2, z1 = z.chunk(2, dim=1)
        elif "checkerboard" in self.mode:
            n_dims = z.dim()
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z.size(n_dims - i))]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z.size(n_dims - i))]
            cb = cb1 if "inv" in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(len(z), *((n_dims - 1) * [1]))
            cb = cb.to(z.device)
            z_size = z.size()
            z1 = z.reshape(-1)[torch.nonzero(cb.view(-1), as_tuple=False)].view(
                *z_size[:-1], -1
            )
            z2 = z.reshape(-1)[torch.nonzero((1 - cb).view(-1), as_tuple=False)].view(
                *z_size[:-1], -1
            )
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return [z1, z2], log_det

    def inverse(self, z):
        z1, z2 = z
        if self.mode == "channel":
            z = torch.cat([z1, z2], 1)
        elif self.mode == "channel_inv":
            z = torch.cat([z2, z1], 1)
        elif "checkerboard" in self.mode:
            n_dims = z1.dim()
            z_size = list(z1.size())
            z_size[-1] *= 2
            cb0 = 0
            cb1 = 1
            for i in range(1, n_dims):
                cb0_ = cb0
                cb1_ = cb1
                cb0 = [cb0_ if j % 2 == 0 else cb1_ for j in range(z_size[n_dims - i])]
                cb1 = [cb1_ if j % 2 == 0 else cb0_ for j in range(z_size[n_dims - i])]
            cb = cb1 if "inv" in self.mode else cb0
            cb = torch.tensor(cb)[None].repeat(z_size[0], *((n_dims - 1) * [1]))
            cb = cb.to(z1.device)
            z1 = z1[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z2 = z2[..., None].repeat(*(n_dims * [1]), 2).view(*z_size[:-1], -1)
            z = cb * z1 + (1 - cb) * z2
        else:
            raise NotImplementedError("Mode " + self.mode + " is not implemented.")
        log_det = 0
        return z, log_det


class Merge(Split):
    """
    Same as Split but with forward and backward pass interchanged
    """

    def __init__(self, mode="channel"):
        super().__init__(mode)

    def forward(self, z):
        z_out, log_det = super().inverse(z)
        return z_out, log_det

    def inverse(self, z):
        z_out, log_det = super().forward(z)
        return z_out, log_det


class Squeeze2d(Flow):
    """
    Squeeze operation of multi-scale architecture (RealNVP / Glow),
    which trades spatial resolution for more channels.
    """

    def __init__(self):
        super().__init__()

    def forward(self, z):
        log_det = 0
        s = z.size()
        z = z.view(s[0], s[1] // 4, 2, 2, s[2], s[3])
        z = z.permute(0, 1, 4, 2, 5, 3).contiguous()
        z = z.view(s[0], s[1] // 4, 2 * s[2], 2 * s[3])
        return z, log_det

    def inverse(self, z):
        log_det = 0
        s = z.size()
        z = z.view(s[0], s[1], s[2] // 2, 2, s[3] // 2, 2)
        z = z.permute(0, 1, 3, 5, 2, 4).contiguous()
        z = z.view(s[0], 4 * s[1], s[2] // 2, s[3] // 2)
        return z, log_det

class Squeeze3d(Flow):
    """
    Squeeze operation for 3D images, extending the concept from Glow.
    """

    def __init__(self):
        """
        Constructor
        """
        super().__init__()

    def forward(self, z):
        log_det = 0
        s = z.size()
        z = z.view(s[0], s[1] // 8, 2, 2, 2, s[2], s[3], s[4])
        z = z.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
        z = z.view(s[0], s[1] // 8, s[2] * 2, s[3] * 2, s[4] * 2)
        return z, log_det

    def inverse(self, z):
        log_det = 0
        s = z.size()
        z = z.view(s[0], s[1], s[2] // 2, 2, s[3] // 2, 2, s[4] // 2, 2)
        z = z.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous()
        z = z.view(s[0], 8 * s[1], s[2] // 2, s[3] // 2, s[4] // 2)
        return z, log_det
    
