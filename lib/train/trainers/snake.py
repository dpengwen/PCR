import torch.nn as nn
from lib.utils import net_utils
import torch
from lib.config import cfg
from lib.utils.snake import snake_config

DEBUG = False
class NetworkWrapper(nn.Module):
    def __init__(self, net):
        super(NetworkWrapper, self).__init__()

        self.net = net

        self.ct_crit = net_utils.FocalLoss()
        self.wh_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.reg_crit = net_utils.IndL1Loss1d('smooth_l1')
        self.ex_crit = torch.nn.functional.smooth_l1_loss
        self.py_crit = torch.nn.functional.smooth_l1_loss
        if cfg.poly_cls_branch:
            self.py_cls_crit = torch.nn.functional.cross_entropy
        if cfg.dist_constraint:
            self.dist_constraint_crit = net_utils.DistanceConstraint()
        
    def forward(self, batch):
        output = self.net(batch['inp'], batch)
        scalar_stats = {}
        loss = 0

        ct_loss = self.ct_crit(net_utils.sigmoid(output['ct_hm']), batch['ct_hm'])
        scalar_stats.update({'ct_loss': ct_loss})
        loss += ct_loss

        wh_loss = self.wh_crit(output['wh'], batch['wh'], batch['ct_ind'], batch['ct_01'])
        scalar_stats.update({'wh_loss': wh_loss})
        loss += 0.1 * wh_loss
        
        if cfg.dist_constraint:
            dist_constraint_loss = self.dist_constraint_crit(output['wh'], batch['wh'], batch['ct_ind'], batch['ct_01'])
            scalar_stats.update({'dist_constraint_loss': dist_constraint_loss})
            loss += dist_constraint_loss

        if cfg.region_iou_flag:
            region_iou_loss = self.region_iou_crit(output['wh'], batch['wh'], batch['ct_ind'], batch['ct_01'])
            scalar_stats.update({'region_iou_loss': region_iou_loss})
            loss += 0.5*region_iou_loss

        if cfg.train.ct_whd:
            ct_whd_loss = self.ct_whd_crit(net_utils.sigmoid(output['ct_hm']), batch['ct_hm'])
            scalar_stats.update({'ct_whd_loss': ct_whd_loss})
            loss += 0.01 * ct_whd_loss

        if cfg.train.ct_reg:
            reg_loss = self.reg_crit(output['reg'], batch['reg'], batch['ct_ind'], batch['ct_01'])
            scalar_stats.update({'reg_loss': reg_loss})
            loss += reg_loss

      
        
        if cfg.use_otpg_flag:
            ex_loss = self.ex_crit(output['ex_pred'], output['i_gt_4py'])
            scalar_stats.update({'ex_loss': ex_loss})
            loss += ex_loss

        py_loss = 0
        output['py_pred'] = [output['py_pred'][-1]]
        for i in range(len(output['py_pred'])):
            if cfg.pos_samples_creg:
                N = int(len(output['py_pred'])/2)
                py_loss += self.py_crit(output['py_pred'][i][:N], output['i_gt_py'][:N]) / N
            else:
                py_loss += self.py_crit(output['py_pred'][i], output['i_gt_py']) / len(output['py_pred'])
        scalar_stats.update({'py_loss': py_loss})
        loss += py_loss
        

       
        if cfg.poly_cls_branch:
            pred_poly_cls = output['py_cls'][0]
            gt_poly_cls   = output['poly_cls_labels'].long()
          
            py_cls_loss = 0.0
            py_cls_loss += self.py_cls_crit(pred_poly_cls, gt_poly_cls)
            scalar_stats.update({'py_cls_loss': py_cls_loss})
            loss += py_cls_loss

            #calculating the accuracy
            _, pred_cls_labels = pred_poly_cls.max(1,keepdim=True)
            correct_preds = pred_cls_labels.eq(gt_poly_cls.view_as(pred_cls_labels)).sum()
            accuracy = correct_preds/torch.tensor(1.0*len(output['poly_cls_labels']))
            scalar_stats.update({'poly_cls_acc': accuracy.float()})

        scalar_stats.update({'loss': loss})
        image_stats = {}

        return output, loss, scalar_stats, image_stats

