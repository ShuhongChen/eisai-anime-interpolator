


from _util.util_v0 import * ; import _util.util_v0 as util
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch

import _util.flow_v0 as uflow


class Dataset(torch.utils.data.Dataset):
    def __init__(self, dk, deterministic):
        self.dk = dk
        self.deterministic = deterministic
        self.bns = dk.bns
        self.size = (540, 960)
        return
    def __len__(self):
        return len(self.bns)
    def __getitem__(self, idx, return_more=False):
        s = sf = self.size
        det = self.deterministic
        bn = str(self.bns[idx], encoding='utf-8')
        dk = self.dk

        # read
        x = dk[bn]
        use_flow = x['flows'] is not None
        imgs = torch.stack([i.resize(s).tensor() for i in x['images']])
        if use_flow:
            flows = uflow.flow_resize(
                x['flows'],
                sf,
                mode='bilinear',
            )
        else:
            flows = None

        # augment
        flip = False
        rev = False
        if not det:
            # flip horizontal
            if np.random.rand()<0.5:
                flip = True
                imgs = imgs.flip(dims=(-1,))
                if use_flow:
                    flows = flows.flip(dims=(-1,))
                    flows[:,1] *= -1

            # reverse sequence
            if np.random.rand()<0.5:
                rev = True
                imgs = imgs.flip(dims=(0,))
                if use_flow:
                    flows = flows.flip(dims=(0,))
        
        # package
        ans = {
            'bn': bn,
            'images': imgs,
        }
        if use_flow:
            ans['flows'] = flows
        if return_more:
            ans['flipped'] = flip
            ans['reversed'] = rev
        return ans

from _databacks.rrldextr import DatabackendRRLDExtracted
class Datamodule(pl.LightningDataModule):
    def __init__(self, path, bs, num_workers=4):
        super().__init__()
        self.path = path
        self.bs = bs
        self.dk = DatabackendRRLDExtracted(self.path)
        self.num_workers = num_workers
        return
    def train_dataloader(self):
        ds = Dataset(
            self.dk,
            False,
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.bs,
            shuffle=True, num_workers=self.num_workers,
            drop_last=False,
        )
        return dl
    def val_dataloader(self):
        ds = Dataset(
            self.dk,
            True,
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.bs,
            shuffle=False, num_workers=self.num_workers,
            drop_last=False,
        )
        return dl
    def test_dataloader(self):
        ds = Dataset(
            self.dk,
            True,
        )
        dl = torch.utils.data.DataLoader(
            ds, batch_size=self.bs,
            shuffle=False, num_workers=self.num_workers,
            drop_last=False,
        )
        return dl




