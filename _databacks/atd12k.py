


from _util.util_v0 import * ; import _util.util_v0 as uutil
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch

class DatabackendATD12k:
    def __init__(self):
        self.dn = './_data/atd12k'
        self.test_source = '540p'
        self.bns = np.array(self.get_bns(), dtype=np.string_)
        return
    def __len__(self):
        return len(self.bns)
    def __getitem__(self, idx):
        if isinstance (idx, int):
            bn = str(self.bns[idx], encoding='utf-8')
        elif isinstance(idx, str):
            bn = idx
        else:
            assert 0, f'{idx=} not understood'
        return {
            'bn': bn,
            'images': [
                I(self.get_fn(bn, i))
                for i in range(3)
            ],
            'flows': torch.tensor(load(f'{self.dn}/preprocessed/rfr_540p/{bn}.pkl')).flip(dims=(0,1)),
        }
    def get_fn(self, bn, fidx):
        tt,tid = bn.split('/')
        if tt=='test':
            dn = f'{self.dn}/raw/test_2k_{self.test_source}'
            ext = 'png' if self.test_source=='540p' else 'jpg'
        else:
            dn = f'{self.dn}/raw/train_10k'
            ext = 'jpg'
        return f'{dn}/{tid}/frame{fidx+1}.{ext}'
    def get_bns(self):
        return sorted([
            f'test/{dn}'
            for dn in os.listdir(f'{self.dn}/raw/test_2k_{self.test_source}')
            if os.path.isdir(f'{self.dn}/raw/test_2k_{self.test_source}/{dn}')
        ])




        