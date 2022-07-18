



from _util.util_v0 import * ; import _util.util_v0 as uutil
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch

import _util.flow_v0 as uflow
import _util.distance_transform_v0 as udist
import _util.sketchers_v1 as usketchers
import _train.frame_interpolation.helpers.gridnet_v1 as ugridnet
import _train.frame_interpolation.helpers.interpolator_v0 as uinterpolator


class Resnet(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = ch = channels
        self.net = nn.Sequential(
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
        )
        return
    def forward(self, x):
        return x + self.net(x)
class Synthesizer(nn.Module):
    def __init__(self, size, channels_image, channels_flow, channels_mask, channels_feature):
        super().__init__()
        self.size = size
        self.diam = diam(self.size)
        self.channels_image = cimg = channels_image
        self.channels_flow = cflow = channels_flow
        self.channels_mask = cmask = channels_mask
        self.channels_feature = cfeat = channels_feature
        self.channels = ch = cimg+cflow//2+cmask+cfeat
        self.interpolator = uinterpolator.Interpolator(self.size, mode='bilinear')
        self.net = nn.Sequential(
            nn.Conv2d(ch+3, 64, kernel_size=1, padding=0),
            Resnet(64),
            nn.Sequential(
                nn.PReLU(64),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
            ),
            Resnet(32),
            nn.Sequential(
                nn.PReLU(32),
                nn.Conv2d(32, 16, kernel_size=3, padding=1),
                nn.BatchNorm2d(16),
            ),
            Resnet(16),
            nn.Sequential(
                nn.PReLU(16),
                nn.Conv2d(16, 3, kernel_size=3, padding=1),
            ),
        )
        return
    def forward(self, images, flows, masks, features, return_more=False):
        itp = self.interpolator
        images = [(images[0]+images[1])/2,] + images
        logimgs = [itp(u2d.pixel_logit(i[:,:3])) for i in images]
        cat = torch.cat([
            *logimgs,
            *[itp(f).norm(dim=1, keepdim=True)/self.diam for f in flows],
            *[itp(m) for m in masks],
            *[itp(f) for f in features],
        ], dim=1)
        residual = self.net(cat)
        return torch.sigmoid(logimgs[0] + 0.5*residual), (locals() if return_more else None)

class FlowZMetric(nn.Module):
    def __init__(self):
        super().__init__()
        return
    def forward(self, img0, img1, flow0, flow1, return_more=False):
        # B(i0,f0) = i1
        # B(i1,f1) = i0
        # F(x,f0,z0)
        # F(x,f1,z1)
        img0 = kornia.color.rgb_to_lab(img0[:,:3])
        img1 = kornia.color.rgb_to_lab(img1[:,:3])
        return [
            -0.1*(img1 - uflow.flow_backwarp(img0, flow0)).norm(dim=1, keepdim=True),  # z0
            -0.1*(img0 - uflow.flow_backwarp(img1, flow1)).norm(dim=1, keepdim=True),  # z1
        ], (locals() if return_more else None)
class NEDT(nn.Module):
    def __init__(self):
        super().__init__()
        return
    def forward(
        self, img, t=2.0, sigma_factor=1/540, k=1.6, epsilon=0.01,
        kernel_factor=4, exp_factor=540/15, return_more=False,
    ):
        with torch.no_grad():
            dog = usketchers.batch_dog(
                img, t=t, sigma=img.shape[-2]*sigma_factor, k=k,
                epsilon=epsilon, kernel_factor=kernel_factor, clip=False,
            )
            edt = udist.batch_edt((dog>0.5).float())
            ans = 1 - (-edt*exp_factor / max(edt.shape[-2:])).exp()
        return ans, (locals() if return_more else None)
class HalfWarper(nn.Module):
    def __init__(self):
        super().__init__()
        self.channels_image = 4*3
        self.channels_flow = 2*2
        self.channels_mask = 2*1
        self.channels = self.channels_image + self.channels_flow + self.channels_mask
    def morph_open(self, x, k):
        if k==0:
            return x
        else:
            with torch.no_grad():
                return kornia.morphology.open(x, torch.ones(k,k,device=x.device))
    def forward(self, img0, img1, flow0, flow1, z0, z1, k, t=0.5, return_more=False):
        # forewarps
        flow0_ = (1-t) * flow0
        flow1_ = t * flow1
        f01 = uflow.forewarp(img0, flow1_, mode='sm', metric=z1, mask=True)
        f10 = uflow.forewarp(img1, flow0_, mode='sm', metric=z0, mask=True)
        f01i,f01m = f01[:,:-1], self.morph_open(f01[:,-1:], k=k)
        f10i,f10m = f10[:,:-1], self.morph_open(f10[:,-1:], k=k)
        
        # base guess
        base0 = f01m*f01i + (1-f01m)*f10i
        base1 = f10m*f10i + (1-f10m)*f01i
        ans = [
            [  # images
                base0, base1,
                f01i, f10i,
            ],
            [  # flows
                flow0_, flow1_,
            ],
            [  # masks
                f01m, f10m,
            ],
        ]
        return ans, (locals() if return_more else None)

class ResnetFeatureExtractor(nn.Module):
    def __init__(self, inferserve_query, size_in=None):
        super().__init__()
        self.inferserve_query = iq = inferserve_query
        self.size_in = si = size_in
        if iq[0]=='torchvision':
            # use pytorch pretrained resnet50
            self.base_hparams = None
            resnet = tv.models.resnet50(pretrained=True)

            self.resize = T.Resize(256)
            self.resnet_preprocess = T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
            self.conv1 = resnet.conv1
            self.bn1 = resnet.bn1
            self.relu = resnet.relu      #   64ch, 128p (assuming 256p input)
            self.maxpool = resnet.maxpool
            self.layer1 = resnet.layer1  #  256ch,  64p
            self.layer2 = resnet.layer2  #  512ch,  32p
        else:
            base = userving.infer_model_load(*iq).eval()
            self.base_hparams = base.hparams
            
            self.resize = T.Resize(base.hparams.largs.size)
            self.resnet_preprocess = base.resnet_preprocess
            self.conv1 = base.resnet.conv1
            self.bn1 = base.resnet.bn1
            self.relu = base.resnet.relu      #   64ch, 128p (assuming 256p input)
            self.maxpool = base.resnet.maxpool
            self.layer1 = base.resnet.layer1  #  256ch,  64p
            self.layer2 = base.resnet.layer2  #  512ch,  32p
        if self.size_in is None:
            self.sizes_out = None
        else:
            s = self.resize.size
            self.sizes_out = [
                pixel_ij(rescale_dry(si, (s//2)/si[0]), rounding='ceil'),  # conv1, 128p
                pixel_ij(rescale_dry(si, (s//4)/si[0]), rounding='ceil'),  # layer1, 64p
                pixel_ij(rescale_dry(si, (s//8)/si[0]), rounding='ceil'),  # layer2, 32p
            ]
        self.channels = [
            64,
            256,
            512,
        ]
        return
    def forward(self, x, force_sizes_out=False, return_more=False):
        ans = []
        x = x[:,:3]
        x = self.resize(x)
        x = self.resnet_preprocess(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        ans.append(x)  # conv1
        x = self.maxpool(x)
        x = self.layer1(x)
        ans.append(x)  # layer1
        x = self.layer2(x)
        ans.append(x)  # layer2
        if force_sizes_out or (self.sizes_out is None):
            self.sizes_out = [tuple(q.shape[-2:]) for q in ans]
        return ans, (locals() if return_more else None)

class NetNedt(nn.Module):
    def __init__(self):
        super().__init__()
        chin = 3+1+4+4+1+1
        ch = 16
        chout = 1
        self.net = nn.Sequential(
            nn.PReLU(chin),
            nn.Conv2d(chin, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, chout, kernel_size=3, padding=1),
        )
        return
    def forward(self, out_base, out_base_nedt, hw_imgs, hw_masks, return_more=False):
        cat = torch.cat([
            out_base,  # 3
            out_base_nedt,  # 1
            hw_imgs[0],  # 4
            hw_imgs[1],  # 4
            hw_masks[0],  # 1
            hw_masks[1],  # 1
        ], dim=1)
        log = u2d.pixel_logit(cat.clip(0,1))
        ans = torch.sigmoid(self.net(log))
        return ans, (locals() if return_more else None)
class NetTail(nn.Module):
    def __init__(self):
        super().__init__()
        chin = 3+1+1
        ch = 16
        chout = 3
        self.net = nn.Sequential(
            nn.PReLU(chin),
            nn.Conv2d(chin, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch),
            nn.PReLU(ch),
            nn.Conv2d(ch, chout, kernel_size=3, padding=1),
        )
        return
    def forward(self, out_base, out_base_nedt, pred_nedt, return_more=False):
        cat = torch.cat([
            out_base,  # 3
            out_base_nedt,  # 1
            pred_nedt,  # 1
        ], dim=1)
        log = u2d.pixel_logit(cat.clip(0,1))
        ans = torch.sigmoid(log[:,:3] + self.net(log))
        return ans, (locals() if return_more else None)



class SoftsplatLite(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = ResnetFeatureExtractor(
            ('torchvision', 'resnet50'),
            (540, 960),
        )
        self.z_metric = FlowZMetric()
        self.flow_downsamplers = [
            uinterpolator.Interpolator(s, mode='bilinear')
            for s in self.feature_extractor.sizes_out
        ]
        self.gridnet_converter = ugridnet.GridnetConverter(
            self.feature_extractor.channels,
            [32,64,128],
        )
        self.gridnet = ugridnet.Gridnet(
            *[32,64,128],
            total_dropout_p=0.0,
            depth=1,  # equivalent to u-net
        )
        self.nedt = NEDT()
        self.half_warper = HalfWarper()
        self.synthesizer = Synthesizer(
            (540, 960),
            self.half_warper.channels_image,
            self.half_warper.channels_flow,
            self.half_warper.channels_mask,
            self.gridnet.channels_0,
        )
        return
    def forward(self, x, t=0.5, k=5, return_more=False):
        rm = return_more
        flow0,flow1 = x['flows'].swapaxes(0,1)
        img0,img1 = x['images'][:,0], x['images'][:,-1]
        (z0,z1),locs_z = self.z_metric(img0, img1, flow0, flow1, return_more=rm)
        img0 = torch.cat([img0, self.nedt(img0)[0]], dim=1)
        img1 = torch.cat([img1, self.nedt(img1)[0]], dim=1)

        # images and flows
        (hw_imgs,hw_flows,hw_masks),locs_hw = self.half_warper(
            img0, img1, flow0, flow1, z0, z1, k, t=t, return_more=rm,
        )

        # features
        feats0,locs_fe0 = self.feature_extractor(img0, return_more=rm)
        feats1,locs_fe1 = self.feature_extractor(img1, return_more=rm)
        warps = []
        for ft0,ft1,ds in zip(feats0, feats1, self.flow_downsamplers):
            (w,_,_),_ = self.half_warper(
                ft0,ft1, ds(flow0,1),ds(flow1,1), ds(z0),ds(z1), k, t=t,
            )
            warps.append((w[0]+w[1])/2)
        feats = self.gridnet(self.gridnet_converter(warps))

        # synthesis
        pred,locs_synth = self.synthesizer(
            hw_imgs, hw_flows, hw_masks, [feats[0],], return_more=rm,
        )
        return pred, (locals() if rm else None)

class DTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.net_nedt = NetNedt()
        self.net_tail = NetTail()
        self.nedt = NEDT()
        return
    def forward(self, x, out_base, locs_base, return_more=False):
        rm = return_more
        with torch.no_grad():
            out_base_nedt,locs_base_nedt = self.nedt(out_base, return_more=rm)
        hw_imgs,hw_masks = locs_base['hw_imgs'], locs_base['hw_masks']
        pred_nedt,locs_nedt = self.net_nedt(out_base, out_base_nedt, hw_imgs, hw_masks, return_more=rm)
        pred,locs_tail = self.net_tail(out_base, out_base_nedt, pred_nedt.clone().detach(), return_more=rm)
        return torch.cat([pred, pred_nedt], dim=1), (locals() if rm else None)









