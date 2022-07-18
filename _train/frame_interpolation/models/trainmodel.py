


from _util.util_v0 import * ; import _util.util_v0 as util
from _util.twodee_v0 import * ; import _util.twodee_v0 as u2d
from _util.pytorch_v0 import * ; import _util.pytorch_v0 as utorch

import _util.flow_v0 as uflow
import _util.distance_transform_v0 as udist
import _util.sketchers_v1 as usketchers

import _train.frame_interpolation.models.ssldtm as ssldtm
class TrainModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        # setup networks
        self.ssl = ssldtm.SoftsplatLite()
        self.dtm = ssldtm.DTM()

        # losses and metrics
        self.metrics_train, self.metrics_val, lwargs = self.get_metrics()
        self.loss_lpips = utorch.LPIPSLoss(net_type='alex')
        self.loss_nedt = utorch.LaplacianPyramidLoss(n_levels=3, colorspace=None, mode='l1')
        return
    def get_metrics(self):
        _kwargs = {
            'dist_sync_on_step': True,
        }
        _lwargs = {
            't': 2.0,
            'sigma': 1.0,
            **_kwargs,
        }
        _lwargs_eval = {
            't': 2.0,
            'sigma': 1.0,
            **_kwargs,
        }
        train = torchmetrics.MetricCollection({
            'train_psnr': utorch.PSNRMetric(data_range=1.0, **_kwargs),
            'train_ssim': utorch.SSIMMetric(**_kwargs),
            'train_lineratio': usketchers.LineRatioMetric(**_lwargs),
            'train_hausdorff': udist.HausdorffDistance2dMetric(**_lwargs),
            'train_chamfer_t': udist.ChamferDistance2dTMetric(**_lwargs),
        })
        val = torchmetrics.MetricCollection({
            'val_psnr': utorch.PSNRMetric(data_range=1.0, **_kwargs),
            'val_ssim': utorch.SSIMMetric(**_kwargs),
            'val_lineratio': usketchers.LineRatioMetric(**_lwargs_eval),
            'val_hausdorff': udist.HausdorffDistance2dMetric(**_lwargs_eval),
            'val_chamfer_t': udist.ChamferDistance2dTMetric(**_lwargs_eval),
            'val_chamfer_p': udist.ChamferDistance2dPMetric(**_lwargs_eval),
            'val_chamfer': udist.ChamferDistance2dMetric(**_lwargs_eval),
            'val_lpips': utorch.LPIPSMetric(net_type='alex', **_kwargs),
        })
        return train, val, _lwargs_eval
    def loss(self, pred, gt, return_more=False):
        pred,pred_nedt = pred[:,:3], pred[:,3:]
        lp = self.loss_lpips(pred, gt)
        gt_nedt,_ = self.ssl.nedt(gt)
        ldt = self.loss_nedt(pred_nedt, gt_nedt)
        return {
            'loss': 30*lp + 5*ldt,
            'loss_lpips': lp,
            'loss_dt': ldt,
        }
    def forward(self, x, t=0.5, return_more=False):
        out_ssl,_ = self.ssl(x, t=t, return_more=True)
        out_dtm,_ = self.dtm(x, out_ssl, _)
        return out_dtm, (locals() if return_more else None)

    def training_step(self, batch, batch_idx):
        gt = batch['images'][:,1]
        pred,_ = self.forward(batch, return_more=False)
        loss = self.loss(pred, gt, return_more=False)
        loss_reduced = {k: v.mean() for k,v in loss.items()}
        mets = self.metrics_train(pred[:,:3], gt)
        
        # log
        for k,v in loss_reduced.items():
            self.log(f'train_{k}', v, sync_dist=True)
        for k,v in mets.items():
            self.log(k, v, on_step=True, on_epoch=False, sync_dist=True)
        return {
            'loss': loss_reduced['loss'],
        }
    def validation_step(self, batch, batch_idx):
        gt = batch['images'][:,1]
        pred,_ = self.forward(batch, return_more=False)
        loss = self.loss(pred, gt, return_more=False)
        loss_reduced = {k: v.mean() for k,v in loss.items()}
        mets = self.metrics_val(pred[:,:3], gt)
        
        # log
        for k,v in loss_reduced.items():
            self.log(f'val_{k}', v, sync_dist=True)
        for k,v in mets.items():
            self.log(k, v, on_step=False, on_epoch=True, sync_dist=True)
        return {
            'val_loss': loss_reduced['loss'],
        }

    def configure_optimizers(self):
        tunable = [
            self.ssl.gridnet,
            self.ssl.gridnet_converter,
            self.ssl.synthesizer,
            self.dtm,
        ] + ([] if True else [self.ssl.feature_extractor])
        opt = torch.optim.Adam(
            [
                param
                for net in tunable
                for param in net.parameters()
            ],
            lr=0.001,
        )
        sched = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
            ),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1,
            'strict': True,
        }
        return [opt,], [sched,]



