


from _util.util_v0 import * ; import _util.util_v0 as util
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch
from _util.video_v0 import * ; import _util.video_v0 as uvid

import _util.flow_v0 as uflow
import _util.frames_v0 as uframes

device = torch.device('cuda')


# hparams
size = 360, 640
bs = 12

th_pan_minavg = 0.25  # low: more disq
th_pan_maxvar = 0.25  # high: more disq
th_pan_maxcov = 0.5  # low: more disq
th_cov_minmov = 2.0  # high: more disq
th_cov_mincov = 0.05**2  # high: more disq
th_rlin_maxrlin = 0.3  # low: more disq

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('video', type=str)
parser.add_argument('duplicates', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()


# check completion
fn_out = mkfile(args.output)

# helpers
outbnd = lambda f: (f<0).any(dim=1) | (f[:,0]>size[0]) | (f[:,1]>size[1])
mg = torch.stack(torch.meshgrid([
    torch.arange(size[0]),
    torch.arange(size[1]),
])).to(device)[None,]

# load model
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
model = RAFT().eval().to(device)

# load video
vr = uvid.VideoReaderDALISeq(
    args.video,
    start=0,
    # stop=1000,
    step=1,
    size=size,
    bs=bs,
)
dups = uframes.DatabaseVideoFrameDuplicates(
    args.duplicates
)
vre = uvid.VideoReaderDALIExclusion(vr, dups, bs=bs)

# loop
quals = []
imgs_prev = torch.zeros(2, 3, *size, device=device)
frs_prev = torch.tensor([-2, -1], device=device)
for batch in tqdm(vre):
    if len(batch['images'])<2: continue

    # resize to flow processing size
    imgs = torch.cat([imgs_prev, batch['images']])
    frs = torch.cat([frs_prev, batch['frames']])
    qual = torch.ones(len(batch['images']), dtype=torch.bool, device=device)
    qual = qual & (frs[:-2]>0) & (frs[1:-1]>0) & (frs[2:]>0)

    # update prevs
    imgs_prev = imgs[-2:]
    frs_prev = frs[-2:]

    # disqualify end credits
    qual = qual & ((imgs[1:-1]<2/255).all(dim=1).float().mean(dim=(1,2))<0.8)
    if (~qual).all(): continue

    # calc heuristic flows
    a0 = imgs[:-2]
    a2 = imgs[2:]
    b = imgs[1:-1]
    with torch.no_grad():
        f01 = fa = model(a0, b)[0]
        f12 = fb = model(a2, b)[0]
        # f01 = fa = uflownet2.flownet2_forward(torch.stack([a0,b], dim=1), model)
        # f12 = fb = uflownet2.flownet2_forward(torch.stack([a2,b], dim=1), model)
    na,nb = fa.norm(dim=1), fb.norm(dim=1)
    fgv = outbnd(mg+fa) | outbnd(mg+fb)
    rawcov = (na>th_cov_minmov) | (nb>th_cov_minmov)
    cov = (rawcov & ~fgv).float()
    covsum = cov.sum(dim=(1,2))

    # disqualify pans
    qual = qual & ~((
            covsum>(th_pan_maxcov*size[0]*size[1])
        ) | (
            (na.mean(dim=(1,2))>th_pan_minavg)
            & (fa.std(dim=(2,3)).min(dim=1).values<th_pan_maxvar)
        ) | (
            (nb.mean(dim=(1,2))>th_pan_minavg)
            & (fb.std(dim=(2,3)).min(dim=1).values<th_pan_maxvar)
    ))

    # disqualify indeterminate linear
    qual = qual & (covsum>(th_cov_mincov*size[0]*size[1]))

    # disqualify high relative linear discrepancy
    rellin = ((fa+fb)/2).norm(dim=1) / (fa-fb).norm(dim=1)
    rellins = (rellin*cov).sum(dim=(1,2)) / covsum
    qual = qual & (rellins<th_rlin_maxrlin)

    # qualify
    if (~qual).all(): continue
    ifrs = frs.int().cpu().numpy()
    qual = [
        [ifrs[i:i+3], rl, cs]
        for i,(q,rl,cs) in enumerate(zip(
            qual,
            rellins.cpu().numpy(),
            (covsum/(size[0]*size[1])).cpu().numpy(),
        )) if q
    ]
    quals.extend(qual)

    # if len(quals)>10: break

# save output
write(
    'frames,linearity,coverage\n' +
    '\n'.join([
        ','.join([
            '-'.join([f'{f:06d}' for f in fs]),
            f'{rl:.16f}',
            f'{cs:.16f}',
        ])
        for fs,rl,cs in quals
    ]),
    mkfile(fn_out),
)

