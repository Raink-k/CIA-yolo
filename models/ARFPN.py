import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ARFPN']



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))




class LiteRepConv(nn.Module):
    """
    训练：3×3 + 1×1
    推理：等效单一 3×3
    """
    def __init__(self, c1, c2):
        super().__init__()
        self.deploy = False

        self.conv3 = Conv(c1, c2, 3)
        self.conv1 = Conv(c1, c2, 1)

    def forward(self, x):
        if self.deploy:
            return self.rep_conv(x)
        return self.conv3(x) + self.conv1(x)

    def _fuse_conv_bn(self, conv, bn):
        w = conv.weight
        mean = bn.running_mean
        var = bn.running_var
        gamma = bn.weight
        beta = bn.bias
        eps = bn.eps

        std = torch.sqrt(var + eps)
        w_fused = w * (gamma / std).reshape(-1, 1, 1, 1)
        b_fused = beta - mean * gamma / std
        return w_fused, b_fused

    def switch_to_deploy(self):
        if self.deploy:
            return

        w3, b3 = self._fuse_conv_bn(self.conv3.conv, self.conv3.bn)
        w1, b1 = self._fuse_conv_bn(self.conv1.conv, self.conv1.bn)

        # 1×1 padding → 3×3
        w1_pad = torch.zeros_like(w3)
        w1_pad[:, :, 1:2, 1:2] = w1

        weight = w3 + w1_pad
        bias = b3 + b1

        self.rep_conv = nn.Conv2d(
            self.conv3.conv.in_channels,
            self.conv3.conv.out_channels,
            kernel_size=3,
            stride=self.conv3.conv.stride,
            padding=self.conv3.conv.padding,
            bias=True
        )

        self.rep_conv.weight.data = weight
        self.rep_conv.bias.data = bias

        del self.conv3
        del self.conv1

        self.deploy = True



class LiteRepC3(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        c_ = int(c2 * e)

        self.cv1 = Conv(c1, c_, 1)
        self.cv2 = Conv(c1, c_, 1)
        self.m = nn.Sequential(*[LiteRepConv(c_, c_) for _ in range(n)])
        self.cv3 = Conv(c_, c2, 1)

    def forward(self, x):
        return self.cv3(self.m(self.cv1(x)) + self.cv2(x))

    def switch_to_deploy(self):
        for m in self.m:
            m.switch_to_deploy()




class SimpleBiFPN(nn.Module):
    def __init__(self, length):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(length, dtype=torch.float32), requires_grad=True)
        self.swish = Swish()
        self.epsilon = 1e-4

    def forward(self, x):
        weights = self.weight / (torch.sum(self.swish(self.weight), dim=0) + self.epsilon)
        weighted = [weights[i] * x[i] for i in range(len(x))]
        stacked = torch.stack(weighted, dim=0)
        return torch.sum(stacked, dim=0)




class ARFPN(nn.Module):
    def __init__(self, in_channels_list, out_channels, n_layers=1):
        super().__init__()

        self.rep_blocks = nn.ModuleList([
            LiteRepC3(c1, out_channels, n=n_layers)
            for c1 in in_channels_list
        ])

        self.fuse = SimpleBiFPN(len(in_channels_list))

    def forward(self, feats):
        reps = [block(x) for block, x in zip(self.rep_blocks, feats)]
        return self.fuse(reps)

    def switch_to_deploy(self):
        for block in self.rep_blocks:
            block.switch_to_deploy()
