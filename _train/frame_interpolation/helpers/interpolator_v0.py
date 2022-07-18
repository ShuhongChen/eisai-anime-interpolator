


from _util.util_v0 import * ; import _util.util_v0 as util
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch


class Interpolator(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.size = size
        self.mode = mode
        return
    def forward(self, x, is_flow=False):
        if x.shape[-2]==self.size:
            return x
        if len(x.shape)==4:
            # bs,ch,h,w
            bs,ch,h,w = x.shape
            ans = nn.functional.interpolate(
                x,
                size=self.size,
                mode=self.mode,
                align_corners=(False,None)[self.mode=='nearest'],
            )
            if is_flow:
                ans = ans * torch.tensor(
                    [b/a for a,b in zip((h,w), self.size)],
                    device=ans.device,
                )[None,:,None,None]
            return ans
        elif len(x.shape)==5:
            # bs,k,ch,h,w (merge bs and k)
            bs,k,ch,h,w = x.shape
            return self.forward(
                x.view(bs*k,ch,h,w),
                is_flow=is_flow,
            ).view(bs,k,ch,*self.size)
        else:
            assert 0





