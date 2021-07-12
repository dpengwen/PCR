import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg
from lib.csrc.roi_align_layer.roi_align import ROIAlign

DEBUG = False
class CircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4):
        super(CircConv, self).__init__()
        self.n_adj = n_adj
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1)

    def forward(self, input, adj):
        input = torch.cat([input[..., -self.n_adj:], input, input[..., :self.n_adj]], dim=2)
        return self.fc(input)

class DilatedCircConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(DilatedCircConv, self).__init__()

        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1, dilation=self.dilation)

    def forward(self, input, adj=None):
        if self.n_adj != 0:
            input = torch.cat([input[..., -self.n_adj*self.dilation:], input, input[..., :self.n_adj*self.dilation]], dim=2)
            
        return self.fc(input)

_conv_factory = {
    'grid': CircConv,
    'dgrid': DilatedCircConv
}
class BasicBlock(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1):
        super(BasicBlock, self).__init__()

        self.conv = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation)
        self.relu = nn.ReLU(inplace=True)
        self.norm = nn.BatchNorm1d(out_state_dim)

    def forward(self, x, adj=None):
        x = self.conv(x, adj)
        x = self.relu(x)
        x = self.norm(x)
        return x

class ClsNet(nn.Module):
    def __init__(self, input_dim=128, roi_h=7, roi_w=7):
        super(ClsNet, self).__init__()
        self.pooler = ROIAlign((roi_h, roi_w))
        self.convs = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=1, padding=0, bias=False),
            nn.ReLU(inplace=True)
        )
        # self.fc = nn.Sequential(
        #     nn.Linear(64*roi_h*roi_w,1024),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(1024, 1024),
        #     nn.ReLU(inplace=True)
        # )
        self.avgpool = nn.AvgPool2d(roi_h)
        self.fc_layers = nn.Sequential(
            nn.Linear(128,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,128),
            nn.ReLU(inplace=True)
        )

        self.cls_fc = nn.Linear(128, 2)

    def prepare_training(self, cnn_feature, output, batch):
        w = cnn_feature.size(3)
        xs = (batch['act_ind'] % w).float()[..., None]
        ys = (batch['act_ind'] // w).float()[..., None]
        wh = batch['awh']
        bboxes = torch.cat([xs - wh[..., 0:1] / 2,
                            ys - wh[..., 1:2] / 2,
                            xs + wh[..., 0:1] / 2,
                            ys + wh[..., 1:2] / 2], dim=2)
        rois = rcnn_snake_utils.box_to_roi(bboxes, batch['act_01'].byte())
        roi = self.pooler(cnn_feature, rois)
        return roi
    def prepare_testing(self, cnn_feature, output):
        if rcnn_snake_config.nms_ct:
            detection, ind = self.nms_abox(output)
        else:
            ind = output['detection'][..., 4] > rcnn_snake_config.ct_score
            detection = output['detection'][ind]
            ind = torch.cat([torch.full([ind[i].sum()], i) for i in range(len(ind))], dim=0)

        ind = ind.to(cnn_feature.device)
        abox = detection[:, :4]
        roi = torch.cat([ind[:, None], abox], dim=1)

        roi = self.pooler(cnn_feature, roi)
        output.update({'detection': detection, 'roi_ind': ind})

        return roi

    def preparing_hbb(self, polys, poly_inds):
        xmin = torch.min(polys[:,:,0], dim=-1)[0]
        ymin = torch.min(polys[:,:,1], dim=-1)[0]
        xmax = torch.max(polys[:,:,0], dim=-1)[0]
        ymax = torch.max(polys[:,:,1], dim=-1)[0]
        hbbs = torch.cat([xmin[...,None],ymin[...,None],xmax[...,None],ymax[...,None]], dim=-1)
        rois = torch.cat([poly_inds[:,None], hbbs], dim=-1)
        if 0:
            import matplotlib.pyplot as plt 
            import numpy 
            import numpy as np 
            x = polys.detach().cpu().numpy()
            y = hbbs.detach().cpu().numpy()
            I = np.zeros((128,128))
            
            for k in range(len(x)):
                plt.imshow(I)
                ply = x[k]
                hbb = y[k]
                xmin,ymin,xmax,ymax = hbb
                hbb_pts = np.array([xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax,xmin,ymin]).reshape(-1,2)
                plt.plot(ply[:,0], ply[:,1],'r')
                plt.plot(hbb_pts[:,0], hbb_pts[:,1], 'g+--')
                plt.show()
                plt.close()
        return rois

    def forward(self, cnn_feature, polys, poly_inds, batch=None):
        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                rois = self.preparing_hbb(polys, poly_inds)
            if DEBUG:
                print("rois.shape:", rois.shape)
                print("cnn_feature.shape:", cnn_feature.shape)
            roi_feat = self.pooler(cnn_feature, rois)
            if DEBUG:
                print("roi_feat.shape:", roi_feat.shape)
            conv_feat = self.convs(roi_feat)
            if DEBUG:
                print("conv_feat.shape:", conv_feat.shape)
            feat = self.avgpool(conv_feat)
            feat = self.fc_layers(feat.view(feat.shape[0],-1))
            cls_output = self.cls_fc(feat)
            if DEBUG:
                print("cls_output.shape:", cls_output.shape)
                exit()
            return cls_output
        if not self.training:
            if 0:
                print("poly_inds:", poly_inds)
                print("polys:", polys.device)
                exit()
            with torch.no_grad():
                rois = self.preparing_hbb(polys, poly_inds.to(polys.device))
                roi_feat = self.pooler(cnn_feature, rois)
                conv_feat = self.convs(roi_feat)
                feat  = self.avgpool(conv_feat)
                feat = self.fc_layers(feat.view(feat.shape[0],-1))
                cls_output = self.cls_fc(feat)
                soft_cls_output = F.softmax(cls_output, dim=-1)
            return soft_cls_output