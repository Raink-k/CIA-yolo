import torch
import torch.nn as nn

# =========================================================
#                    MSDA MODULE
# =========================================================

__all__ = ['MultiDilatelocalAttention', 'C2f_IMSD_Block']


class DilateAttention(nn.Module):
    """Dilated Local Attention"""

    def __init__(self, head_dim, qk_scale=None, attn_drop=0.,
                 kernel_size=3, dilation=1):
        super().__init__()
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5
        self.kernel_size = kernel_size
        self.unfold = nn.Unfold(
            kernel_size=kernel_size,
            dilation=dilation,
            padding=dilation * (kernel_size - 1) // 2,
            stride=1
        )
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, q, k, v):
        B, d, H, W = q.shape

        q = q.reshape(
            B, d // self.head_dim, self.head_dim, 1, H * W
        ).permute(0, 1, 4, 3, 2)  # B,h,N,1,d

        k = self.unfold(k).reshape(
            B, d // self.head_dim, self.head_dim,
            self.kernel_size * self.kernel_size, H * W
        ).permute(0, 1, 4, 2, 3)  # B,h,N,d,k*k

        attn = (q @ k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        v = self.unfold(v).reshape(
            B, d // self.head_dim, self.head_dim,
            self.kernel_size * self.kernel_size, H * W
        ).permute(0, 1, 4, 3, 2)  # B,h,N,k*k,d

        x = (attn @ v).transpose(1, 2).reshape(B, H, W, d)
        return x


class MultiDilatelocalAttention(nn.Module):
    """Multi-Scale Dilated Local Attention"""

    def __init__(self, dim, num_heads=8, qkv_bias=True,
                 qk_scale=None, attn_drop=0., proj_drop=0.,
                 kernel_size=3, dilation=(1, 2, 3, 4)):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.dilation = dilation
        self.kernel_size = kernel_size

        head_dim = dim // num_heads
        self.num_dilation = len(dilation)
        assert num_heads % self.num_dilation == 0

        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=qkv_bias)

        self.dilate_attention = nn.ModuleList([
            DilateAttention(
                head_dim=head_dim,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                kernel_size=kernel_size,
                dilation=d
            ) for d in dilation
        ])

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape

        qkv = self.qkv(x).reshape(
            B, 3, self.num_dilation,
            C // self.num_dilation, H, W
        ).permute(2, 1, 0, 3, 4, 5)

        y = x.reshape(
            B, self.num_dilation, C // self.num_dilation, H, W
        ).permute(1, 0, 3, 4, 2)

        for i in range(self.num_dilation):
            y[i] = self.dilate_attention[i](
                qkv[i][0], qkv[i][1], qkv[i][2]
            )

        y = y.permute(1, 2, 3, 0, 4).reshape(B, H, W, C)
        y = self.proj(y)
        y = self.proj_drop(y).permute(0, 3, 1, 2)
        return y


# =========================================================
#                    MSBLOCK MODULE
# =========================================================

def autopad(k, p=None, d=1):
    if d > 1:
        k = d * (k - 1) + 1
    if p is None:
        p = k // 2
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(
            c1, c2, k, s, autopad(k, p, d),
            groups=g, dilation=d, bias=False
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MSBlockLayer(nn.Module):
    def __init__(self, inc, ouc, k):
        super().__init__()
        self.in_conv = Conv(inc, ouc, 1)
        self.mid_conv = Conv(ouc, ouc, k, g=ouc)
        self.out_conv = Conv(ouc, inc, 1)

    def forward(self, x):
        return self.out_conv(self.mid_conv(self.in_conv(x)))


# =========================================================
#             MSBLOCK + MSDA FUSION MODULE
# =========================================================

class MSBlock_MSDA(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes,
                 in_expand_ratio=3., mid_expand_ratio=2.,
                 layers_num=3, in_down_ratio=2.,
                 msda_heads=8):
        super().__init__()

        in_channel = int(inc * in_expand_ratio // in_down_ratio)
        self.mid_channel = in_channel // len(kernel_sizes)
        groups = int(self.mid_channel * mid_expand_ratio)

        self.in_conv = Conv(inc, in_channel)
        self.mid_convs = nn.ModuleList()
        self.attentions = nn.ModuleList()

        for k in kernel_sizes:
            if k == 1:
                self.mid_convs.append(nn.Identity())
                self.attentions.append(None)
            else:
                self.mid_convs.append(
                    nn.Sequential(*[
                        MSBlockLayer(self.mid_channel, groups, k)
                        for _ in range(int(layers_num))
                    ])
                )
                self.attentions.append(
                    MultiDilatelocalAttention(
                        dim=self.mid_channel,
                        num_heads=msda_heads
                    )
                )

        self.out_conv = Conv(in_channel, ouc, 1)

    def forward(self, x):
        out = self.in_conv(x)
        outputs = []

        for i, (conv, attn) in enumerate(zip(self.mid_convs, self.attentions)):
            xi = out[:, i * self.mid_channel:(i + 1) * self.mid_channel]

            if i > 0:
                xi = xi + outputs[i - 1]

            xi = conv(xi)

            if attn is not None:
                xi = attn(xi)

            outputs.append(xi)

        out = torch.cat(outputs, dim=1)
        return self.out_conv(out)


# =========================================================
#                 C2f + MSBlock_MSDA
# =========================================================

class C2f_IMSD_Block(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)

        self.cv1 = Conv(c1, 2 * self.c, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            MSBlock_MSDA(self.c, self.c, kernel_sizes=[1, 3, 3])
            for _ in range(n)
        )

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))
