


from _util.util_v0 import * ; import _util.util_v0 as util
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch


class Gridnet(nn.Module):
    def __init__(self, channels_0, channels_1, channels_2, total_dropout_p, depth):
        super().__init__()
        self.channels_0 = ch0 = channels_0
        self.channels_1 = ch1 = channels_1
        self.channels_2 = ch2 = channels_2
        self.total_dropout_p = p = total_dropout_p
        self.depth = depth
        self.encoders = nn.ModuleList([GridnetEncoder(ch0,ch1,ch2) for i in range(self.depth)])
        self.decoders = nn.ModuleList([GridnetDecoder(ch0,ch1,ch2) for i in range(self.depth)])
        self.total_dropout = GridnetTotalDropout(p)
        return
    def forward(self, x):
        for e,enc in enumerate(self.encoders):
            t = [self.total_dropout(i) for i in t] if e!=0 else x
            t = enc(t)
        for d,dec in enumerate(self.decoders):
            t = [self.total_dropout(i) for i in t]
            t = dec(t)
        return t
class GridnetEncoder(nn.Module):
    def __init__(self, channels_0, channels_1, channels_2):
        super().__init__()
        self.channels_0 = ch0 = channels_0
        self.channels_1 = ch1 = channels_1
        self.channels_2 = ch2 = channels_2
        self.resnet_0 = GridnetResnet(ch0)
        self.resnet_1 = GridnetResnet(ch1)
        self.resnet_2 = GridnetResnet(ch2)
        self.downsample_01 = GridnetDownsample(ch0, ch1)
        self.downsample_12 = GridnetDownsample(ch1, ch2)
        return
    def forward(self, x):
        out = [None,]*3
        out[0] = self.resnet_0(x[0])
        out[1] = self.resnet_1(x[1]) + self.downsample_01(out[0])
        out[2] = self.resnet_2(x[2]) + self.downsample_12(out[1])
        return out
class GridnetDecoder(nn.Module):
    def __init__(self, channels_0, channels_1, channels_2):
        super().__init__()
        self.channels_0 = ch0 = channels_0
        self.channels_1 = ch1 = channels_1
        self.channels_2 = ch2 = channels_2
        self.resnet_0 = GridnetResnet(ch0)
        self.resnet_1 = GridnetResnet(ch1)
        self.resnet_2 = GridnetResnet(ch2)
        self.upsample_10 = GridnetUpsample(ch1, ch0)
        self.upsample_21 = GridnetUpsample(ch2, ch1)
        return
    def forward(self, x):
        out = [None,]*3
        out[2] = self.resnet_2(x[2])
        out[1] = self.resnet_1(x[1]) + self.upsample_21(out[2])
        out[0] = self.resnet_0(x[0]) + self.upsample_10(out[1])
        return out
class GridnetConverter(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in = cin = channels_in
        self.channels_out = cout = channels_out
        self.nets = nn.ModuleList([
            nn.Sequential(
                nn.PReLU(a),
                nn.Conv2d(a, b, kernel_size=1, padding=0),
                nn.BatchNorm2d(b),
            )
            for a,b in zip(cin, cout)
        ])
        return
    def forward(self, x):
        return [m(q) for m,q in zip(self.nets, x)]


class GridnetResnet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = ch = channels
        self.net = nn.Sequential(
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
        )
        return
    def forward(self, x):
        return x + self.net(x)
class GridnetDownsample(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in = chin = channels_in
        self.channels_out = chout = channels_out
        self.net = nn.Sequential(
            nn.PReLU(chin),
            nn.Conv2d(chin, chin, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(chin),
            nn.PReLU(chin),
            nn.Conv2d(chin, chout, kernel_size=3, padding=1),
            nn.BatchNorm2d(chout),
        )
        return
    def forward(self, x):
        return self.net(x)
class GridnetUpsample(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()
        self.channels_in = chin = channels_in
        self.channels_out = chout = channels_out
        self.net = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.PReLU(chin),
            nn.Conv2d(chin, chout, kernel_size=3, padding=1),
            nn.BatchNorm2d(chout),
            nn.PReLU(chout),
            nn.Conv2d(chout, chout, kernel_size=3, padding=1),
            nn.BatchNorm2d(chout),
        )
        return
    def forward(self, x):
        return self.net(x)
class GridnetTotalDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        assert 0<=p<1
        self.p = p
        self.weight = 1/(1-p)
        return
    def get_drop(self, x):
        d = torch.rand(len(x))[:,None,None,None]<self.p
        d = (1-d.float()).to(x.device) * self.weight
        return d
    def forward(self, x, force_drop=None):
        if force_drop is True:
            ans = x * self.get_drop(x)
        elif force_drop is False:
            ans = x
        else:
            if self.training:
                ans = x * self.get_drop(x)
            else:
                ans = x
        return ans







