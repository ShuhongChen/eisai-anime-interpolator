


from _util.util_v0 import * ; import _util.util_v0 as util
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch
from _util.video_v0 import * ; import _util.video_v0 as uvid

device = torch.device('cuda')


# hparams
bs = 24
size = (540, 960)

parser = argparse.ArgumentParser()
parser.add_argument('video', type=str)
parser.add_argument('rrld', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()


# setup
bns = [
    line.split(',')[0]
    for line in read(args.rrld).split('\n')[1:]
    if line
]
odn = mkdir(args.output)
todo = sorted(set(bns))

# prep bns
vfn = args.video
vm = uvid.video_metadata(vfn)
sbn = [
    tuple(int(f) for f in frs.split('-'))
    for frs in todo
]#[:10]
incl = set([j for i in sbn for j in i])
excl = set(range(vm['frame_count'])) - incl
ofns = defaultdict(list)
for tri in sbn:
    fstr = '-'.join([f'{f:06d}' for f in tri])
    for fr in tri:
        ofns[fr].append(f'{args.output}/{fstr}/{fr:06d}.png')
ofns = dict(ofns)

# load video
vr = uvid.VideoReaderDALISeq(
    vfn,
    bs=bs,
    start=min(incl),
    stop=max(incl)+1,
    size=size,
)
vre = uvid.VideoReaderDALIExclusion(
    vr, excl, bs=bs,
)

# save images
for batch in tqdm(vre):
    imgs,frs = batch['images'].cpu(), batch['frames'].int().cpu().numpy()
    for i,f in zip(imgs, frs):
        i = I(i)
        for ofn in ofns[f]:
            i.save(mkfile(ofn))
del vr, vre
gc.collect()




