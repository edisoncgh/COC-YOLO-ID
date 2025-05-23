import torch
import torch.nn as nn

__all__ = ['L_GSConv']

def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Mish() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class L_GSConv(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def shuffle(self, x, group_size):
        b, n, h, w = x.size()
        x = x.reshape(b, n // group_size, group_size, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        return x.reshape(b, n, h, w)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = self.cv2(x1)
        
        x1_shuff = self.shuffle(x1, group_size=4)
        x2_shuff = self.shuffle(x2, group_size=4)
        return torch.cat((x1_shuff, x2_shuff), 1)