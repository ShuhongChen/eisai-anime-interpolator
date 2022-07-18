


try:
    import torch
    import torch.nn as nn
except:
    pass

try:
    import torchvision as tv
    import torchvision.transforms as T
    import torchvision.transforms.functional as F
except:
    pass

try:
    import pytorch_lightning as pl
except:
    pass
try:
    import torchmetrics
    import lpips
except:
    from argparse import Namespace
    torchmetrics = Namespace(Metric=object)
try:
    import wandb
except:
    pass

try:
    import kornia
except:
    pass

try:
    import detectron2
    from detectron2 import model_zoo as _
    from detectron2 import engine as _
    from detectron2 import config as _
    from detectron2 import data as _
    from detectron2.utils import visualizer as _
except:
    pass

try:
    from nvidia import dali
    from nvidia.dali.plugin import pytorch as _
except:
    pass

try:
    import cupy
except:
    pass

try:
    import skimage
    from skimage import measure as _
    from skimage import color as _
    from skimage import segmentation as _
    from skimage import filters as _
    from scipy.spatial.transform import Rotation
except:
    pass


#################### UTILITIES ####################

try:
    # @cupy.memoize(for_each_device=True)
    def cupy_launch(func, kernel):
        return cupy.cuda.compile_with_cache(kernel).get_function(func)
except:
    cupy_launch = lambda func,kernel: None

def reset_parameters(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    return model

def channel_squeeze(x, dim=1):
    a = x.shape[:dim]
    b = x.shape[dim+2:]
    return x.reshape(*a, -1, *b)
def channel_unsqueeze(x, shape, dim=1):
    a = x.shape[:dim]
    b = x.shape[dim+1:]
    return x.reshape(*a, *shape, *b)

def default_collate(items, device=None):
    return to(dict(torch.utils.data.dataloader.default_collate(items)), device)
def to(x, device):
    if device is None:
        return x
    if issubclass(x.__class__, dict):
        return dict({
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k,v in x.items()
        })
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, np.ndarray):
        return torch.tensor(x).to(device)
    assert 0, 'data not understood'

#################### LOSSES + METRICS ####################

class SSIMMetric(torchmetrics.Metric):
    # torchmetrics has memory leak
    def __init__(self, window_size=11, **kwargs):
        super().__init__(**kwargs)
        self.window_size = window_size
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ans = kornia.losses.ssim(target, preds, self.window_size).mean((1,2,3))
        self.running_sum += ans.sum()
        self.running_count += len(ans)
        return
    def compute(self):
        return self.running_sum.float() / self.running_count
class SSIMMetricCPU(torchmetrics.Metric):
    full_state_update=False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ans = [
            skimage.metrics.structural_similarity(
                p.permute(1,2,0).cpu().numpy(),
                t.permute(1,2,0).cpu().numpy(),
                multichannel=True,
                gaussian=True,
                # data_range=255,
            )
            for p,t in zip(preds, target)
        ]
        self.running_sum += sum(ans)
        self.running_count += len(ans)
        return
    def compute(self):
        return self.running_sum / self.running_count

class PSNRMetric(torchmetrics.Metric):
    # torchmetrics averages samples before taking log
    def __init__(self, data_range=1.0, **kwargs):
        super().__init__(**kwargs)
        self.data_range = torch.tensor(data_range)
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ans = -10 * torch.log10( (target-preds).pow(2).mean((1,2,3)) )
        self.running_sum += 20*torch.log10(self.data_range) + ans.sum()
        self.running_count += len(ans)
        return
    def compute(self):
        return self.running_sum.float() / self.running_count
class PSNRMetricCPU(torchmetrics.Metric):
    full_state_update=False
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        ans = [
            skimage.metrics.peak_signal_noise_ratio(
                p.permute(1,2,0).cpu().numpy(),
                t.permute(1,2,0).cpu().numpy(),
                # data_range=255,
            )
            for p,t in zip(preds, target)
        ]
        self.running_sum += sum(ans)
        self.running_count += len(ans)
        return
    def compute(self):
        return self.running_sum / self.running_count

class LPIPSMetric(torchmetrics.Metric):
    full_state_update=False
    def __init__(self, net_type='alex', **kwargs):
        super().__init__(**kwargs)
        self.net_type = net_type
        assert self.net_type in ['alex', 'vgg', 'squeeze']
        self.model = lpips.LPIPS(net=self.net_type)
        self.add_state('running_sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('running_count', default=torch.tensor(0.0), dist_reduce_fx='sum')
        return
    def update(self, preds: torch.Tensor, target: torch.Tensor):
        if preds.requires_grad:
            ans = self.model(preds, target).mean((1,2,3))
        else:
            with torch.no_grad():
                ans = self.model(preds, target).mean((1,2,3))
        self.running_sum += ans.sum()
        self.running_count += len(ans)
        return
    def compute(self):
        return self.running_sum.float() / self.running_count
class LPIPSLoss(nn.Module):
    def __init__(self, net_type='alex', **kwargs):
        super().__init__()
        self.net_type = net_type
        assert self.net_type in ['alex', 'vgg', 'squeeze']
        self.model = lpips.LPIPS(net=self.net_type, **kwargs)
        return
    def forward(self, preds: torch.Tensor, target: torch.Tensor):
        ans = self.model(preds, target).mean((1,2,3))
        return ans

class LaplacianPyramidLoss(nn.Module):
    def __init__(self, n_levels=3, colorspace=None, mode='l1'):
        super().__init__()
        self.n_levels = n_levels
        self.colorspace = colorspace
        self.mode = mode
        assert self.mode in ['l1', 'l2']
        return
    def forward(self, preds, target, force_levels=None, force_mode=None):
        if self.colorspace=='lab':
            preds = kornia.color.rgb_to_lab(preds.float())
            target = kornia.color.rgb_to_lab(target.float())
        lvls = self.n_levels if force_levels==None else force_levels
        preds = kornia.geometry.transform.build_pyramid(preds, lvls)
        target = kornia.geometry.transform.build_pyramid(target, lvls)
        mode = self.mode if force_mode==None else force_mode
        if mode=='l1':
            ans = torch.stack([
                (p-t).abs().mean((1,2,3))
                for p,t in zip(preds,target)
            ]).mean(0)
        elif mode=='l2':
            ans = torch.stack([
                (p-t).norm(dim=1, keepdim=True).mean((1,2,3))
                for p,t in zip(preds,target)
            ]).mean(0)
        else:
            assert 0
        return ans



