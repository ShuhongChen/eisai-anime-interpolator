


from _util.util_v0 import * ; import _util.util_v0 as uutil
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch
import _util.distance_transform_v0 as udist

import _train.frame_interpolation.models.ssldtm as models

device = torch.device('cuda')


# raft helper
from _train.frame_interpolation.helpers.raft_v1 import rfr_new as uraft
class RAFT(nn.Module):
    def __init__(self, path='./checkpoints/anime_interp_full.ckpt'):
        super().__init__()
        self.raft = uraft.RFR(Namespace(
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

# wrapper
def interpolate(raft, ssl, dtm, img0, img1, t=0.5, return_more=False):
    with torch.no_grad():
        img0,img1 = img0.t()[None,:3].to(device), img1.t()[None,:3].to(device)
        flow0,_ = raft(img0, img1)
        flow1,_ = raft(img1, img0)
        x = {
            'images': torch.stack([img0,img1], dim=1),
            'flows': torch.stack([flow0,flow1], dim=1),
        }
        out_ssl,_ = ssl(x, t=t, return_more=True)
        out_dtm,_ = dtm(x, out_ssl, _, return_more=return_more)
        ans = I(out_dtm[0,:3])
    return ans, (locals() if return_more else None)

# evaluate
if __name__=='__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('img0', type=str)
    ap.add_argument('img1', type=str)
    ap.add_argument('--fps', type=int, default=12)
    ap.add_argument('--out', type=str, default='./temp/interpolation')
    args = ap.parse_args()

    # load images
    img0 = I(args.img0).convert('RGB')
    img1 = I(args.img1).convert('RGB')
    assert img0.size==img1.size
    original_size = img0.size
    img0 = img0.resize((540,960))
    img1 = img1.resize((540,960))

    # load models
    ssl = models.SoftsplatLite()
    dtm = models.DTM()
    ssl.load_state_dict(torch.load('./checkpoints/ssl.pt'))
    dtm.load_state_dict(torch.load('./checkpoints/dtm.pt'))
    ssl = ssl.to(device).eval()
    dtm = dtm.to(device).eval()
    raft = RAFT().eval().to(device)

    # interpolate
    n = args.fps + 1
    ts = np.linspace(0,1,n)[1:-1]
    uutil.mkdir(args.out)
    img0.resize(original_size).save(f'{args.out}/{n-1:02d}_{0:02d}.png')
    img1.resize(original_size).save(f'{args.out}/{n-1:02d}_{n-1:02d}.png')
    for i,t in enumerate(ts):
        ans,_ = interpolate(raft, ssl, dtm, img0, img1, t=t)
        ans.resize(original_size).save(f'{args.out}/{n-1:02d}_{i+1:02d}.png')



