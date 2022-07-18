


from _util.util_v0 import * ; import _util.util_v0 as util
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch

import _util.sketchers_v1 as usketchers


############### DISTANCE TRANSFORM ###############

# img tensor: (bs,h,w) or (bs,1,h,w)
# returns same shape
# expects white lines, black whitespace
# defaults to diameter if empty image
_batch_edt_kernel = ('kernel_dt', '''
    extern "C" __global__ void kernel_dt(
        const int bs,
        const int h,
        const int w,
        const float diam2,
        float* data,
        float* output
    ) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= bs*h*w) {
            return;
        }
        int pb = idx / (h*w);
        int pi = (idx - h*w*pb) / w;
        int pj = (idx - h*w*pb - w*pi);

        float cost;
        float mincost = diam2;
        for (int j = 0; j < w; j++) {
            cost = data[h*w*pb + w*pi + j] + (pj-j)*(pj-j);
            if (cost < mincost) {
                mincost = cost;
            }
        }
        output[idx] = mincost;
        return;
    }
''')
_batch_edt = None
def batch_edt(img, block=1024):
    # must initialize cuda/cupy after forking
    global _batch_edt
    if _batch_edt is None:
        _batch_edt = cupy_launch(*_batch_edt_kernel)

    # bookkeeppingg
    if len(img.shape)==4:
        assert img.shape[1]==1
        img = img.squeeze(1)
        expand = True
    else:
        expand = False
    bs,h,w = img.shape
    diam2 = h**2 + w**2
    odtype = img.dtype
    grid = (img.nelement()+block-1) // block

    # cupy implementation
    if img.is_cuda:
        # first pass, y-axis
        data = ((1-img.type(torch.float32)) * diam2).contiguous()
        intermed = torch.zeros_like(data)
        _batch_edt(
            grid=(grid, 1, 1),
            block=(block, 1, 1),  # < 1024
            args=[
                cupy.int32(bs),
                cupy.int32(h),
                cupy.int32(w),
                cupy.float32(diam2),
                data.data_ptr(),
                intermed.data_ptr(),
            ],
        )
        
        # second pass, x-axis
        intermed = intermed.permute(0,2,1).contiguous()
        out = torch.zeros_like(intermed)
        _batch_edt(
            grid=(grid, 1, 1),
            block=(block, 1, 1),
            args=[
                cupy.int32(bs),
                cupy.int32(w),
                cupy.int32(h),
                cupy.float32(diam2),
                intermed.data_ptr(),
                out.data_ptr(),
            ],
        )
        ans = out.permute(0,2,1).sqrt()
        ans = ans.type(odtype) if odtype!=ans.dtype else ans
    
    # default to scipy cpu implementation
    else:
        sums = img.sum(dim=(1,2))
        ans = torch.tensor(np.stack([
            scipy.ndimage.morphology.distance_transform_edt(i)
            if s!=0 else  # change scipy behavior for empty image
            np.ones_like(i) * np.sqrt(diam2)
            for i,s in zip(1-img, sums)
        ]), dtype=odtype)

    if expand:
        ans = ans.unsqueeze(1)
    return ans


############### DERIVED DISTANCES ###############

# input: (bs,h,w) or (bs,1,h,w)
# returns: (bs,)
# normalized s.t. metric is same across proportional image scales

# average of two asymmetric distances
# normalized by diameter and area
def batch_chamfer_distance(gt, pred, block=1024, return_more=False):
    t = batch_chamfer_distance_t(gt, pred, block=block)
    p = batch_chamfer_distance_p(gt, pred, block=block)
    cd = (t + p) / 2
    return cd
def batch_chamfer_distance_t(gt, pred, block=1024, return_more=False):
    assert gt.device==pred.device and gt.shape==pred.shape
    bs,h,w = gt.shape[0], gt.shape[-2], gt.shape[-1]
    dpred = batch_edt(pred, block=block)
    cd = (gt*dpred).float().mean((-2,-1)) / np.sqrt(h**2+w**2)
    if len(cd.shape)==2:
        assert cd.shape[1]==1
        cd = cd.squeeze(1)
    return cd
def batch_chamfer_distance_p(gt, pred, block=1024, return_more=False):
    assert gt.device==pred.device and gt.shape==pred.shape
    bs,h,w = gt.shape[0], gt.shape[-2], gt.shape[-1]
    dgt = batch_edt(gt, block=block)
    cd = (pred*dgt).float().mean((-2,-1)) / np.sqrt(h**2+w**2)
    if len(cd.shape)==2:
        assert cd.shape[1]==1
        cd = cd.squeeze(1)
    return cd

# normalized by diameter
# always between [0,1]
def batch_hausdorff_distance(gt, pred, block=1024, return_more=False):
    assert gt.device==pred.device and gt.shape==pred.shape
    bs,h,w = gt.shape[0], gt.shape[-2], gt.shape[-1]
    dgt = batch_edt(gt, block=block)
    dpred = batch_edt(pred, block=block)
    hd = torch.stack([
        (dgt*pred).amax(dim=(-2,-1)),
        (dpred*gt).amax(dim=(-2,-1)),
    ]).amax(dim=0).float() / np.sqrt(h**2+w**2)
    if len(hd.shape)==2:
        assert hd.shape[1]==1
        hd = hd.squeeze(1)
    return hd


############### TORCHMETRICS ###############

class ChamferDistance2dMetric(torchmetrics.Metric):
    full_state_update=False
    def __init__(
            self, block=1024, convert_dog=True,
            t=2.0, sigma=1.0, k=1.6, epsilon=0.01, kernel_factor=4, clip=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.block = block
        self.convert_dog = convert_dog
        self.dog_params = {
            't': t, 'sigma': sigma, 'k': k, 'epsilon': epsilon,
            'kernel_factor': kernel_factor, 'clip': clip,
        }
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.convert_dog:
            preds = (usketchers.batch_dog(preds, **self.dog_params)>0.5).float()
            target = (usketchers.batch_dog(target, **self.dog_params)>0.5).float()
        dist = batch_chamfer_distance(target, preds, block=self.block)
        self.running_sum += dist.sum()
        self.running_count += len(dist)
        return
    def compute(self):
        return self.running_sum.float() / self.running_count
class ChamferDistance2dTMetric(ChamferDistance2dMetric):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.convert_dog:
            preds = (usketchers.batch_dog(preds, **self.dog_params)>0.5).float()
            target = (usketchers.batch_dog(target, **self.dog_params)>0.5).float()
        dist = batch_chamfer_distance_t(target, preds, block=self.block)
        self.running_sum += dist.sum()
        self.running_count += len(dist)
        return
class ChamferDistance2dPMetric(ChamferDistance2dMetric):
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.convert_dog:
            preds = (usketchers.batch_dog(preds, **self.dog_params)>0.5).float()
            target = (usketchers.batch_dog(target, **self.dog_params)>0.5).float()
        dist = batch_chamfer_distance_p(target, preds, block=self.block)
        self.running_sum += dist.sum()
        self.running_count += len(dist)
        return

class HausdorffDistance2dMetric(torchmetrics.Metric):
    def __init__(
            self, block=1024, convert_dog=True,
            t=2.0, sigma=1.0, k=1.6, epsilon=0.01, kernel_factor=4, clip=False,
            **kwargs,
        ):
        super().__init__(**kwargs)
        self.block = block
        self.convert_dog = convert_dog
        self.dog_params = {
            't': t, 'sigma': sigma, 'k': k, 'epsilon': epsilon,
            'kernel_factor': kernel_factor, 'clip': clip,
        }
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if self.convert_dog:
            preds = (usketchers.batch_dog(preds, **self.dog_params)>0.5).float()
            target = (usketchers.batch_dog(target, **self.dog_params)>0.5).float()
        dist = batch_hausdorff_distance(target, preds, block=self.block)
        self.running_sum += dist.sum()
        self.running_count += len(dist)
        return
    def compute(self):
        return self.running_sum.float() / self.running_count






