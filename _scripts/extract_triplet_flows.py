


from _util.util_v0 import * ; import _util.util_v0 as uutil
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch

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

# parse args
parser = argparse.ArgumentParser()
parser.add_argument('images', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

# loop
raft = RAFT().to(device).eval()
odn = mkdir(args.output)
for dn in tqdm(sorted(os.listdir(args.images))):
    if not os.path.isdir(f'{args.images}/{dn}'): continue
    ofn = f'{odn}/{dn}.pt'
    if os.path.isfile(ofn): continue
    try:
        fr0,frT,fr1 = dn.split('-')
        img0 = I(f'{args.images}/{dn}/{fr0}.png')
        img1 = I(f'{args.images}/{dn}/{fr1}.png')
    except:
        continue
    img0,img1 = img0.t()[None,:3].to(device), img1.t()[None,:3].to(device)
    with torch.no_grad():
        flow0,_ = raft(img0, img1)
        flow1,_ = raft(img1, img0)
    flows = torch.stack([flow0,flow1], dim=1)[0].cpu()
    torch.save(flows, ofn)



