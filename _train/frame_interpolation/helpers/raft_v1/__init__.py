


from _util.util_v0 import * ; import _util.util_v0 as uutil
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch
from . import rfr_new


class RAFT(nn.Module):
    def __init__(self, path='./_data/atd12k/checkpoints/anime_interp_full.ckpt'):
        super().__init__()
        self.raft = rfr_new.RFR(Namespace(
            small=False,
            mixed_precision=False,
        ))
        if path is not None:
            sd = torch.load(path)['model_state_dict']
            self.raft.load_state_dict({
                k[len('module.flownet.'):]: v
                for k,v in sd.items()
                if k.startswith('module.flownet.')
            }, strict=False)
        return
    def forward(self, img0, img1, flow0=None, iters=12, return_more=False):
        if flow0 is not None:
            flow0 = flow0.flip(dims=(1,))
        out = self.raft(img1, img0, iters=iters, flow_init=flow0)
        return out[0].flip(dims=(1,)), (locals() if return_more else None)


