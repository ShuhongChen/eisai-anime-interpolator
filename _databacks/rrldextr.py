


from _util.util_v0 import * ; import _util.util_v0 as uutil
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch

class DatabackendRRLDExtracted:
    def __init__(self, path):
        self.path = path
        bns = set(os.listdir(f'{self.path}/images')).intersection(
            set([fn.split('.')[0] for fn in os.listdir(f'{self.path}/flows')])
        )
        bns = sorted(list(bns))
        self.bns = np.array(bns, dtype=np.string_)
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
                I(f'{self.path}/images/{bn}/{fr}.png')
                for fr in bn.split('-')
            ],
            'flows': torch.load(f'{self.path}/flows/{bn}.pt'),
        }




        