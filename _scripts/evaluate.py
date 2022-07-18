


from _util.util_v0 import * ; import _util.util_v0 as uutil
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch
import _util.distance_transform_v0 as udist

import _train.frame_interpolation.models.ssldtm as models
import _databacks.atd12k as datasets

device = torch.device('cuda')


# load data
dk = datasets.DatabackendATD12k()
bns_test = sorted([str(bn, encoding='utf-8') for bn in dk.bns if bn.startswith(b'test/')])
assert len(bns_test)==2000, 'missing ATD test data'

# load models and metrics
ssl = models.SoftsplatLite()
dtm = models.DTM()
ssl.load_state_dict(torch.load('./checkpoints/ssl.pt'))
dtm.load_state_dict(torch.load('./checkpoints/dtm.pt'))
ssl = ssl.to(device).eval()
dtm = dtm.to(device).eval()
metrics = torchmetrics.MetricCollection({
    'psnr': utorch.PSNRMetricCPU(),
    'ssim': utorch.SSIMMetricCPU(),
    'lpips': utorch.LPIPSMetric(net_type='alex'),
    'chamfer': udist.ChamferDistance2dMetric(t=2.0, sigma=1.0),
    # 'chamfer': udist.ChamferDistance2dMetric(block=512, t=2.0, sigma=1.0),
}).to(device).eval()

# evaluate
results = defaultdict(list)
for bn in tqdm(bns_test):
    x = dk[bn]
    x = {
        'images': torch.stack([
            i.tensor() for i in x['images']
        ], dim=0)[None,].to(device),
        'flows': x['flows'][None,].to(device)
    }
    
    # compute model outputs
    with torch.no_grad():
        out_ssl,_ = ssl(x, return_more=True)
        out_dtm,_ = dtm(x, out_ssl, _)
    pred = out_dtm[:,:3]
    # dump((x,pred), '/dev/shm/test.pkl')
    # exit(0)
    
    # evaluate
    img_gt = x['images'][:,1]
    with torch.no_grad():
        out = metrics(pred, img_gt)
    for k,v in out.items():
        results[k].append(v.item())
    # break

# print results
print(Table([
    ['subset::l', 'metric::l', 'score::l'],
    ['=::>',],
    ['all::l', 'lpips::l', (np.mean(results['lpips']), 'r:.4E')],
    ['all::l', 'chamfer::l', (np.mean(results['chamfer']), 'r:.4E')],
    ['all::l', 'psnr::l', (np.mean(results['psnr']), 'r:.2f')],
    ['all::l', 'ssim::l', (100*np.mean(results['ssim']), 'r:.2f')],
    ['east::l', 'lpips::l', (np.mean([v for bn,v in zip(bns_test,results['lpips']) if 'Japan_' in bn]), 'r:.4E')],
    ['east::l', 'chamfer::l', (np.mean([v for bn,v in zip(bns_test,results['chamfer']) if 'Japan_' in bn]), 'r:.4E')],
    ['west::l', 'lpips::l', (np.mean([v for bn,v in zip(bns_test,results['lpips']) if 'Disney_' in bn]), 'r:.4E')],
    ['west::l', 'chamfer::l', (np.mean([v for bn,v in zip(bns_test,results['chamfer']) if 'Disney_' in bn]), 'r:.4E')],
]))






