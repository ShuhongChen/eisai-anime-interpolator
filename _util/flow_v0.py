


from _util.util_v0 import * ; import _util.util_v0 as util
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch

try:
    import _util.softsplat_v0 as usoftsplat
except:
    pass


####################### WARPING #######################

# expects batched tensors, considered low-level operation
# img: bs, ch, h, w
# flow: bs, xy (pix displace), h, w
def flow_backwarp(img, flow, resample='bilinear', padding_mode='border', align_corners=False):
    if len(img.shape)!=4: img = img[None,]
    if len(flow.shape)!=4: flow = flow[None,]
    q = 2 * flow / torch.tensor([
        flow.shape[-2], flow.shape[-1],
    ], device=flow.device, dtype=torch.float)[None,:,None,None]
    q = q + torch.stack(torch.meshgrid(
        torch.linspace(-1,1, flow.shape[-2]),
        torch.linspace(-1,1, flow.shape[-1]),
    ))[None,].to(flow.device)
    if img.dtype!=q.dtype: img = img.type(q.dtype)

    return nn.functional.grid_sample(
        img,
        q.flip(dims=(1,)).permute(0,2,3,1),
        mode=resample, # nearest, bicubic, bilinear
        padding_mode=padding_mode,  # border, zeros, reflection
        align_corners=align_corners,
    )
backwarp = flow_warp = flow_backwarp

# mode: sum, avg, lin, softmax
# lin/softmax w/out metric defaults to avg
# must use gpu, move back to cpu if retain_device
# typical metric: -20 * | img0 - backwarp(img1,flow) |
def flow_forewarp(img, flow, mode='average', metric=None, mask=False, retain_device=True):
    # setup
    if mode=='sum':
        mode = 'summation'
    elif mode=='avg':
        mode = 'average'
    elif mode in ['lin', 'linear']:
        mode = 'linear' if metric is not None else 'average'
    elif mode in ['sm', 'softmax']:
        mode = 'softmax' if metric is not None else 'average'
    if len(img.shape)!=4: img = img[None,]
    if len(flow.shape)!=4: flow = flow[None,]
    if metric is not None and len(metric.shape)!=4: metric = metric[None,]
    flow = flow.flip(dims=(1,))
    if img.dtype!=torch.float32:
        img = img.type(torch.float32)
    if flow.dtype!=torch.float32:
        flow = flow.type(torch.float32)
    if metric is not None and metric.dtype!=torch.float32:
        metric = metric.type(torch.float32)
    
    # move to gpu if necessary
    assert img.device==flow.device
    if metric is not None: assert img.device==metric.device
    was_cpu = img.device.type=='cpu'
    if was_cpu:
        img = img.to('cuda')
        flow = flow.to('cuda')
        if metric is not None: metric = metric.to('cuda')
    
    # add mask
    if mask:
        bs,ch,h,w = img.shape
        img = torch.cat([
            img,
            torch.ones(bs,1,h,w, dtype=img.dtype, device=img.device)
        ], dim=1)
        
    # forward, move back to cpu if desired
    ans = usoftsplat.FunctionSoftsplat(img, flow, metric, mode)
    if was_cpu and retain_device:
        ans = ans.cpu()
    return ans
forewarp = flow_forewarp

# resizing utility
def flow_resize(flow, size, mode='nearest', align_corners=False):
    # flow: bs,xy,h,w
    size = pixel_ij(size, rounding=True)
    if flow.dtype!=torch.float:
        flow = flow.float()
    if len(flow.shape)==3:
        flow = flow[None,]
    if flow.shape[-2:]==size:
        return flow
    return nn.functional.interpolate(
        flow,
        size=size,
        mode=mode,
        align_corners=align_corners if mode!='nearest' else None,
    ) * torch.tensor(
        [b/a for a,b in zip(flow.shape[-2:],size)],
        device=flow.device,
    )[None,:,None,None]


####################### TRADITIONAL #######################

# dense
_lucaskanade = lambda a,b: np.moveaxis(cv2.optflow.calcOpticalFlowSparseToDense(
        a, b, #grid_step=5, sigma=0.5,
    ), 2, 0)[None,]
_farneback = lambda a,b: np.moveaxis(cv2.calcOpticalFlowFarneback(
        a, b, None, 0.6, 3, 25, 7, 5, 1.2, cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    ), 2, 0)[None,]
_dtvl1_ = cv2.optflow.createOptFlow_DualTVL1()
_dtvl1 = lambda a,b: np.moveaxis(_dtvl1_.calc(
        a, b, None,
    ), 2, 0)[None,]
_simple = lambda a,b: np.moveaxis(cv2.optflow.calcOpticalFlowSF(
        a, b, 3, 5, 5,
    ), 2, 0)[None,]
_pca_ = cv2.optflow.createOptFlow_PCAFlow()
_pca = lambda a,b: np.moveaxis(_pca_.calc(
        a, b, None,
    ), 2, 0)[None,]
_drlof = lambda a,b: np.moveaxis(cv2.optflow.calcOpticalFlowDenseRLOF(
        a, b, None,
    ), 2, 0)[None,]
_deepflow_ = cv2.optflow.createOptFlow_DeepFlow()
_deepflow = lambda a,b: np.moveaxis(_deepflow_.calc(
        a, b, None,
    ), 2, 0)[None,]
def cv2flow(a, b, method='lucaskanade', back=False):
    if method=='lucaskanade':
        f = _lucaskanade
        a = a.convert('L').cv2()
        b = b.convert('L').cv2()
    elif method=='farneback':
        f = _farneback
        a = a.convert('L').cv2()
        b = b.convert('L').cv2()
    elif method=='dtvl1':
        f = _dtvl1
        a = a.convert('L').cv2()
        b = b.convert('L').cv2()
    elif method=='simple':
        f = _simple
        a = a.convert('RGB').cv2()
        b = b.convert('RGB').cv2()
    elif method=='pca':
        f = _pca
        a = a.convert('L').cv2()
        b = b.convert('L').cv2()
    elif method=='drlof':
        f = _drlof
        a = a.convert('RGB').cv2()
        b = b.convert('RGB').cv2()
    elif method=='deepflow':
        f = _deepflow
        a = a.convert('L').cv2()
        b = b.convert('L').cv2()
    else:
        assert 0
    ans = f(b, a)
    if back:
        ans = np.concatenate([
            ans, f(a, b),
        ])
    return torch.tensor(ans).flip(dims=(1,))


####################### FLOWNET2 #######################

def flownet2(img_a, img_b, mode='shm', back=False):
    # package
    url = f'http://localhost:8109/get-flow'
    if mode=='shm':
        t = time.time()
        fn_a = img_a.save(mkfile(f'/dev/shm/_flownet2/{t}/img_a.png'))
        fn_b = img_b.save(mkfile(f'/dev/shm/_flownet2/{t}/img_b.png'))
    elif mode=='net':
        assert False, 'not impl'
        q = u2d.img2uri(img.pil('RGB'))
        data = q.decode()
    resp = requests.get(url, params={
        'img_a': fn_a,
        'img_b': fn_b,
        'mode': mode,
        'back': back,
        # 'vis': vis,
    })
    
    # return
    ans = {'response': resp}
    if resp.status_code==200:
        j = resp.json()
        ans['time'] = j['time']
        ans['output'] = {
            'flow': torch.tensor(load(j['fn_flow'])),
        }
        # if vis:
        #     ans['output']['vis'] = I(j['fn_vis'])
    if mode=='shm':
        shutil.rmtree(f'/dev/shm/_flownet2/{t}')
    return ans


####################### VISUALIZATION #######################

# flow: bs, xy (pix displace), h, w
def flow_vis(flow, radius=0.2):
    assert len(flow.shape)==3 and flow.shape[0]==2
    h,w = flow.shape[-2:]
    m = max(h,w)
    r = radius * m
    hsv = torch.stack([
        torch.atan2(flow[0], flow[1]) + np.pi,
        (torch.norm(flow, dim=0) / r).clip(0, 1),
        torch.ones(h, w, device=flow.device),
    ])
    rgb = kornia.color.hsv_to_rgb(hsv)
    return I(rgb)
def flow_wheel(s=100):
    mg = torch.stack(torch.meshgrid([
        torch.linspace(-1,1,s),
        torch.linspace(-1,1,s),
    ]))
    mg = mg * s
    return flow_vis(mg, radius=1)
fv = flow_vis
fw = flow_wheel









