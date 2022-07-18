


from _util.util_v0 import * ; import _util.util_v0 as util
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch
from _util.video_v0 import * ; import _util.video_v0 as uvid

device = torch.device('cuda')


# params
bs = 2*24
size_frame = (360,640)
write_interval = 10000

# parse args
ap = argparse.ArgumentParser()
ap.add_argument('input', type=str)
ap.add_argument('output', type=str)
args = ap.parse_args()
fn_video = args.input
fn_db = args.output

# setup db
mkfile(fn_db)
with sqlite3.connect(fn_db) as conn:
    c = conn.cursor()
    c.execute("""CREATE TABLE IF NOT EXISTS duplicates (
        frame int,
        rgb_mean float,
        rgb_max float,
        lab_mean float,
        lab_max float,
        psnr float,
        ssim float,
        PRIMARY KEY (frame)
    );""")
with sqlite3.connect(fn_db) as conn:
    c = conn.cursor()
    c.execute("select max(frame) from duplicates;")
    start_frame = c.fetchall()[0][0] or 0
start_frame = start_frame + 1
vmd = video_metadata(fn_video)
if start_frame>=vmd['frame_count']:
    exit(0)

# get first frame
vr0 = VideoReaderDALISeq(
    fn_video,
    bs=1,
    start=start_frame-1,
    stop=start_frame,
    step=1,
    size=size_frame,
)
for x in vr0:
    prev = x['images']
del vr0
    
# process video
vr = VideoReaderDALISeq(
    fn_video,
    bs=bs,
    start=start_frame,
    step=1,
    size=size_frame,
)
todb = []
def _write_todb(todb):
    with sqlite3.connect(fn_db) as conn:
        c = conn.cursor()
        for vals in todb:
            c.execute("""
            INSERT OR REPLACE INTO duplicates (
                frame, rgb_mean, rgb_max, lab_mean, lab_max, psnr, ssim
            ) VALUES (
                :frame, :rgb_mean, :rgb_max, :lab_mean, :lab_max, :psnr, :ssim
            );""", vals)
    return
for x in tqdm(vr, desc=fstrip(fn_video)):
    # batch forward
    imgs = x['images']
    frames = x['frames'].cpu().numpy()
    prevcat = torch.cat([prev, imgs[:-1]])
    diff = (prevcat - imgs).abs()
    diff_max = diff.amax((1,2,3)).cpu().numpy().astype(float)
    diff_mean = diff.mean((1,2,3)).cpu().numpy().astype(float)
    dlab = (
        kornia.color.rgb_to_lab(prevcat) -
        kornia.color.rgb_to_lab(imgs)
    ).norm(dim=1)
    dlab_max = dlab.amax((1,2)).cpu().numpy().astype(float)
    dlab_mean = dlab.mean((1,2)).cpu().numpy().astype(float)
    mse = diff.pow(2).mean((1,2,3))
    psnr = (-10 * torch.log10(mse + 1e-10)).cpu().numpy().astype(float)
    ssim = kornia.losses.ssim(prevcat, imgs, 11).mean((1,2,3)).cpu().numpy().astype(float)
    for i,fr in enumerate(frames):
        todb.append({
            'frame': int(fr),
            'rgb_max': diff_max[i],
            'rgb_mean': diff_mean[i],
            'lab_max': dlab_max[i],
            'lab_mean': dlab_mean[i],
            'psnr': psnr[i],
            'ssim': ssim[i],
        })
    prev = imgs[-1:]

    # update db batch
    if len(todb)>write_interval:
        _write_todb(todb)
        todb = []
#     if frames[-1]>10*24: break
_write_todb(todb)
del vr, x, imgs, prev
gc.collect()



