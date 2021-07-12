import torch
import torch.nn as nn
import torch.nn.functional as F
from .snake import Snake
from .custom_net import ClsNet
from lib.config import cfg
from lib.utils import data_utils
from lib.utils.snake import snake_gcn_utils, snake_config, snake_decode, active_spline


DEBUG = False
class Evolution(nn.Module):
    def __init__(self):
        super(Evolution, self).__init__()
        if cfg.backbone_name == 'resnet50':
            contour_in_feat_channel = cfg.backbone_out_channel
            self.fuse = nn.Conv1d(2*contour_in_feat_channel, contour_in_feat_channel, 1)
        else:
            contour_in_feat_channel = 64
            if cfg.bpoint_feat_enhance == 'aster':
                self.fuse = nn.Conv1d(192, contour_in_feat_channel, 1)
            else:
                self.fuse = nn.Conv1d(128, contour_in_feat_channel, 1)

        if cfg.use_otpg_flag:
            self.init_gcn = Snake(state_dim=128, feature_dim=contour_in_feat_channel+2, conv_type=cfg.snake_conv_type)

        if cfg.csm_version == 'csm_v0':
            self.iter = cfg.evolve_iter
            for i in range(self.iter):
                evolve_gcn = Snake(state_dim=128, feature_dim=contour_in_feat_channel+2, conv_type=cfg.snake_conv_type)
                self.__setattr__('evolve_gcn'+str(i), evolve_gcn)
            final_evolve_gcn = Snake(state_dim=128, feature_dim=contour_in_feat_channel+2, conv_type=cfg.snake_conv_type, snake_type='evolve_final')
            self.__setattr__('evolve_gcn_final', final_evolve_gcn)
        elif cfg.csm_version == 'csm_v1':
            self.iter = cfg.evolve_iter
            for i in range(self.iter):
                evolve_gcn = Snake(state_dim=128, feature_dim=contour_in_feat_channel+2, conv_type=cfg.snake_conv_type)
                self.__setattr__('evolve_gcn'+str(i), evolve_gcn)
            final_loc_evolve = Snake(state_dim=128, feature_dim=contour_in_feat_channel+2, conv_type=cfg.snake_conv_type)
            self.__setattr__('final_loc_evolve', final_loc_evolve)
            
            if cfg.poly_cls_branch:
                final_cls_net = ClsNet(input_dim=128)
                self.__setattr__('final_cls_net', final_cls_net)
        else:
            self.evolve_gcn = Snake(state_dim=128, feature_dim=contour_in_feat_channel+2, conv_type=cfg.snake_conv_type)
            self.iter = cfg.evolve_iter
            for i in range(self.iter):
                if i != self.iter - 1:
                    evolve_gcn = Snake(state_dim=128, feature_dim=contour_in_feat_channel+2, conv_type=cfg.snake_conv_type)
                else:
                    evolve_gcn = Snake(state_dim=128, feature_dim=contour_in_feat_channel+2, conv_type=cfg.snake_conv_type, snake_type='evolve_final')
                self.__setattr__('evolve_gcn'+str(i), evolve_gcn)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def prepare_training(self, output, batch):
        init = snake_gcn_utils.prepare_training(output, batch)
        if cfg.poly_cls_branch:
            spec_num = 80
            init['i_gt_py'] = torch.cat((init['i_gt_py'][:spec_num], init['neg_i_gt_py'][:spec_num]), dim=0)
            init['c_gt_py'] = torch.cat((init['c_gt_py'][:spec_num], init['neg_c_gt_py'][:spec_num]), dim=0)
            init['i_it_py'] = torch.cat((init['i_it_py'][:spec_num], init['neg_i_it_py'][:spec_num]), dim=0)
            init['c_it_py'] = torch.cat((init['c_it_py'][:spec_num], init['neg_c_it_py'][:spec_num]), dim=0)
            init['py_ind']  = torch.cat((init['py_ind'][:spec_num],  init['py_ind'][:spec_num]), dim=0)
            poly_num = len(init['i_gt_py'])
            poly_cls_labels = torch.ones(poly_num)
            poly_cls_labels[int(poly_num/2):] = 0
            output.update({'poly_cls_labels': poly_cls_labels.cuda()})
            
            if 0:
                input_imgs = batch['inp'].detach().cpu().numpy()
                i_gt_py = init['i_gt_py'].detach().cpu().numpy()
                i_it_py = init['i_it_py'].detach().cpu().numpy()
                neg_i_gt_py = init['neg_i_gt_py'].detach().cpu().numpy()
                neg_i_it_py = init['neg_i_it_py'].detach().cpu().numpy()
                py_inds = init['py_ind'].detach().cpu().numpy()

                np.save('input_imgs.npy', input_imgs)
                np.save('i_gt_py.npy', i_gt_py)
                np.save('i_it_py.npy', i_it_py)
                np.save('neg_i_gt_py.npy', neg_i_gt_py)
                np.save('neg_i_it_py.npy', neg_i_it_py)
                np.save('py_inds.npy', py_inds)
                print('saving training data done!')
                exit()

        output.update({'i_it_4py': init['i_it_4py'], 'i_it_py': init['i_it_py']})
        output.update({'i_gt_4py': init['i_gt_4py'], 'i_gt_py': init['i_gt_py']})
        return init

    def prepare_training_evolve(self, output, batch, init):
        evolve = snake_gcn_utils.prepare_training_evolve(output['ex_pred'], init)
        output.update({'i_it_py': evolve['i_it_py'], 'c_it_py': evolve['c_it_py'], 'i_gt_py': evolve['i_gt_py']})
        evolve.update({'py_ind': init['py_ind']})
        return evolve

    def prepare_testing_init(self, output):
        init = snake_gcn_utils.prepare_testing_init(output['detection'][..., :4], output['detection'][..., 4])
        output['detection'] = output['detection'][output['detection'][..., 4] > snake_config.ct_score]
        output.update({'it_ex': init['i_it_4py']})
        return init

    def prepare_testing_evolve(self, output, h, w):
        if cfg.evolve_init == 'poly':
            ex = output['init_poly_pred']
        else:
            ex = output['ex']
        ex[..., 0] = torch.clamp(ex[..., 0], min=0, max=w-1)
        ex[..., 1] = torch.clamp(ex[..., 1], min=0, max=h-1)

        evolve = snake_gcn_utils.prepare_testing_evolve(ex)
        if 0:
            import matplotlib.pyplot as plt 
            import numpy as np 
            I = np.zeros((128,128))
            plt.imshow(I)
            x = ex.cpu().numpy()
            y = evolve['i_it_py'].cpu().numpy()
            for k in range(len(x)):
                xk = x[k]
                yk = y[k]
                yk = np.concatenate((yk,yk[0].reshape(-1,2)),axis=0)
                plt.plot(xk[:,0], xk[:,1], 'go-')
                plt.text(xk[0,0], xk[0,1], '1',color='g')
                plt.text(xk[1,0], xk[1,1], '2',color='g')
                plt.text(xk[2,0], xk[2,1], '3',color='g')
                plt.text(xk[3,0], xk[3,1], '4',color='g')

                plt.plot(yk[:,0], yk[:,1], 'r')
                plt.plot(yk[0,0], yk[0,1], 'yo')
                plt.plot(yk[5,0], yk[5,1], 'y+')
            plt.show()

        output.update({'it_py': evolve['i_it_py']})
        if 0:
            import numpy as np 
            import matplotlib.pyplot as plt 
            print('a:', evolve['i_it_py'].shape)
            x = evolve['i_it_py'].cpu().numpy()
            I = np.zeros((128, 128))
            plt.imshow(I)
            for k in range(len(x)):
                np_ply = x[k]
                plt.plot(np_ply[:,0], np_ply[:,1], 'r')
                plt.plot(np_ply[0,0], np_ply[0,1], 'go')
                plt.plot(np_ply[10,0], np_ply[10,1], 'yo')
            plt.show()
        return evolve
    
    def prepare_testing_evolve_from_hbb(self, output, h, w):
        detection = output['detection']
        if len(detection) == 0:
            i_it_pys = torch.zeros([0, snake_config.poly_num, 2]).to(detection)
            c_it_pys = torch.zeros_like(i_it_pys)
            hbb_corner_pts = torch.zeros([0,4,2]).to(detection)
        else:
            xmin, ymin = detection[..., 0:1], detection[..., 1:2]
            xmax, ymax = detection[..., 2:3], detection[..., 3:4]
            hbb_corner_pts = torch.cat((xmin,ymin,xmin,ymax,xmax,ymax,xmax,ymin),dim=1)
            hbb_corner_pts = hbb_corner_pts.reshape(detection.shape[0], -1, 2)
            hbb_corner_pts[..., 0] = torch.clamp(hbb_corner_pts[..., 0], min=0, max=w-1)
            hbb_corner_pts[..., 1] = torch.clamp(hbb_corner_pts[..., 1], min=0, max=w-1)
            i_it_pys = snake_gcn_utils.uniform_upsample(hbb_corner_pts[None], snake_config.poly_num)[0]
            c_it_pys = snake_gcn_utils.img_poly_to_can_poly(i_it_pys)
        evolve = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys}
        output.update({'it_py': evolve['i_it_py']})
        output.update({'ex': hbb_corner_pts})

        return evolve




    def init_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind):
        if len(i_it_poly) == 0:
            i_4poly = torch.zeros([0, 4, 2]).to(i_it_poly)
            i_poly = torch.zeros([0, snake_config.init_poly_num, 2]).to(i_it_poly)
            return i_4poly, i_poly
            
        if DEBUG:
            print('-----------------')
            print('cnn_feature.shape:', cnn_feature.shape)
            print('i_it_poly.shape:', i_it_poly.shape)
            
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)
        
        center = (torch.min(i_it_poly, dim=1)[0] + torch.max(i_it_poly, dim=1)[0]) * 0.5
        
        ct_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)
        
        if cfg.bpoint_feat_enhance == 'aster':
            edge_feat = init_feature - ct_feature.expand_as(init_feature)
            init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature), edge_feat], dim=1)
            init_feature = self.fuse(init_feature)
        else:
            init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature)], dim=1)
            init_feature = self.fuse(init_feature)

        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        if DEBUG:
            print('adj.shape:', adj.shape)
            print('init_input.shape:', init_input.shape)
            print('i_it_poly.shape:', i_it_poly.shape)
        i_poly = i_it_poly + snake(init_input, adj).permute(0, 2, 1)
        i_4poly = i_poly[:, ::snake_config.init_poly_num//4]

        if DEBUG:
            print('i_poly.shape:', i_poly.shape)
            print("i_4py.shape:", i_4poly.shape)

        return i_4poly, i_poly

    def evolve_poly(self, snake, cnn_feature, i_it_poly, c_it_poly, ind, ply_cls_flag=False):
        if ply_cls_flag and (len(i_it_poly) == 0):
            return torch.empty((0,2)).cuda(), torch.zeros_like(i_it_poly), torch.zeros_like(i_it_poly)
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly),torch.zeros_like(i_it_poly)
        
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)

        if 0:
            print('-----------------Evolving_poly---------------------')
            print('i_it_poly.shape:', i_it_poly.shape)
            print('init_feature.shape:', init_feature.shape)
            print("c_it_poly.shape:", c_it_poly.shape)
            exit()
        
        if cfg.bpoint_feat_enhance == 'aster':
            from lib.utils.snake import snake_text_utils
            center = torch.mean(i_it_poly, dim=1)
            ct_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)
            edge_feat = init_feature - ct_feature.expand_as(init_feature)
            init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature), edge_feat], dim=1)
            init_feature = self.fuse(init_feature)
        else:
            if cfg.evolve_ct_feat:
                center = torch.mean(i_it_poly, dim=1)
                ct_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)
                init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature)], dim=1)
                init_feature = self.fuse(init_feature)
            else:
                pass

        c_it_poly = c_it_poly * snake_config.ro
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)
        
        if ply_cls_flag:
            evolve_polys, evolve_polys_cls = snake(init_input, adj)
            evolve_polys = i_it_poly * snake_config.ro + evolve_polys.permute(0, 2, 1)
            return evolve_polys_cls, evolve_polys, init_feature
        else:
            evolve_py = snake(init_input, adj).permute(0, 2, 1)
            i_poly = i_it_poly * snake_config.ro + evolve_py
            return i_poly, init_feature
    
    def loc_cls_head(self, cnn_feature, loc_snake_net, cls_pred_net, i_it_poly, c_it_poly, ind, batch=None):
        if cfg.poly_cls_branch and (len(i_it_poly) == 0):
            return torch.empty((0,2)).cuda(), torch.zeros_like(i_it_poly),torch.zeros_like(i_it_poly)
        if len(i_it_poly) == 0:
            return torch.zeros_like(i_it_poly),torch.zeros_like(i_it_poly)
        
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        init_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, i_it_poly, ind, h, w)

        if cfg.bpoint_feat_enhance == 'aster':
            from lib.utils.snake import snake_text_utils
            center = torch.mean(i_it_poly, dim=1)
            ct_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)
            edge_feat = init_feature - ct_feature.expand_as(init_feature)
            init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature), edge_feat], dim=1)
            init_feature = self.fuse(init_feature)
        else:
            if cfg.evolve_ct_feat:
                center = torch.mean(i_it_poly, dim=1)
                ct_feature = snake_gcn_utils.get_gcn_feature(cnn_feature, center[:, None], ind, h, w)
                init_feature = torch.cat([init_feature, ct_feature.expand_as(init_feature)], dim=1)
                init_feature = self.fuse(init_feature)
            else:
                pass

        c_it_poly = c_it_poly * snake_config.ro
        init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
        adj = snake_gcn_utils.get_adj_ind(snake_config.adj_num, init_input.size(2), init_input.device)

        loc_offsets = loc_snake_net(init_input, adj).permute(0, 2, 1)
        i_poly = i_it_poly * snake_config.ro + loc_offsets

        if cfg.poly_cls_branch:
            poly_cls = cls_pred_net(cnn_feature, i_it_poly, ind, batch)
        else:
            poly_cls = None
        
        return poly_cls, i_poly, init_feature


    def forward(self, output, cnn_feature, batch=None):
        ret = output
        if batch is not None and 'test' not in batch['meta']:
            with torch.no_grad():
                init = self.prepare_training(output, batch)
            if 0:
                print('out.keys:', output.keys())
                print('i-it-4py.size:', init['i_it_4py'].size())
                print('c-it-4py.size:', init['c_it_4py'].size())
                print('4py-ind.size:', init['4py_ind'])
                print('py-ind.size:', init['py_ind'])
                print('cnn_feat.size:', cnn_feature.size())
                print('i_it_py.size:', init['i_it_py'].size())
                print('c_it_py.size:', init['c_it_py'].size())
                print('i_gt_py.size:', init['i_gt_py'].size())
                print('c_gt_py.size:', init['c_gt_py'].size())
                exit()
            
            if cfg.use_otpg_flag:
                ex_pred, init_poly_pred = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['4py_ind'])
                ret.update({'ex_pred': ex_pred, 'i_gt_4py': output['i_gt_4py'], 'init_poly_pred': init_poly_pred})
            
            # with torch.no_grad():
            #     init = self.prepare_training_evolve(output, batch, init)
            if cfg.csm_version == 'csm_v0':
                py_pred = init['i_it_py']
                c_py_pred = init['c_it_py']
                py_preds = []
                for i in range(self.iter):
                    evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                    py_pred, final_feat  = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'])
                    py_pred = py_pred / snake_config.ro
                    c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred)
                    py_preds.append(py_pred)
                evolve_gcn = self.__getattr__('evolve_gcn_final')
                if cfg.poly_cls_branch:
                    iter_py_cls = []
                    py_cls, py_pred, final_feat = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'], cfg.poly_cls_branch)
                    iter_py_cls.append(py_cls)
                    py_preds.append(py_pred)
                else:
                    py_pred, final_feat  = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'])
                    py_preds.append(py_pred)
            elif cfg.csm_version == 'csm_v1':
                py_pred = init['i_it_py']
                c_py_pred = init['c_it_py']
                py_preds = []
                for i in range(self.iter):
                    evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                    py_pred, final_feat  = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'])
                    py_preds.append(py_pred)
                    py_pred = py_pred / snake_config.ro
                    c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred)
                
                final_loc_evolve = self.__getattr__('final_loc_evolve')
                if cfg.poly_cls_branch:
                    final_cls_net = self.__getattr__('final_cls_net')
                else:
                    final_cls_net = None
                py_cls, py_pred, final_feat = self.loc_cls_head(cnn_feature, final_loc_evolve, final_cls_net, py_pred, c_py_pred, init['py_ind'], batch)
                py_preds.append(py_pred)
                iter_py_cls = [py_cls]
            else:
                if cfg.poly_cls_branch:
                    iter_py_cls = []
                py_pred, final_feat = self.evolve_poly(self.evolve_gcn, cnn_feature, init['i_it_py'], init['c_it_py'], init['py_ind'])
                py_preds = [py_pred]
                for i in range(self.iter):
                    py_pred = py_pred / snake_config.ro
                    c_py_pred = snake_gcn_utils.img_poly_to_can_poly(py_pred)
                    evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                    if (i == self.iter - 1) and cfg.poly_cls_branch:
                        py_cls, py_pred, final_feat = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, 
                                                        init['py_ind'], cfg.poly_cls_branch)
                        iter_py_cls.append(py_cls)
                    else:
                        py_pred, final_feat  = self.evolve_poly(evolve_gcn, cnn_feature, py_pred, c_py_pred, init['py_ind'])
                    py_preds.append(py_pred)

            ret.update({'py_pred': py_preds, 'i_gt_py': output['i_gt_py'] * snake_config.ro})
            
            if cfg.poly_cls_branch:
                ret.update({'py_cls': iter_py_cls})

        if not self.training:
            with torch.no_grad():
                init = self.prepare_testing_init(output)
                
                if cfg.use_otpg_flag:
                    ex_pred, init_poly_pred = self.init_poly(self.init_gcn, cnn_feature, init['i_it_4py'], init['c_it_4py'], init['ind'])
                    ret.update({'ex': ex_pred, 'init_poly_pred': init_poly_pred})
                    evolve = self.prepare_testing_evolve(output, cnn_feature.size(2), cnn_feature.size(3))
                else:
                    evolve = self.prepare_testing_evolve_from_hbb(output, cnn_feature.size(2), cnn_feature.size(3))
                
                if cfg.csm_version == 'csm_v0':
                    py = evolve['i_it_py']
                    c_py = evolve['c_it_py']
                    pys = []
                    for i in range(self.iter):
                        evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                        py, final_feat = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['ind'])
                        py = py / snake_config.ro
                        c_py = snake_gcn_utils.img_poly_to_can_poly(py)
                        pys.append(py)
                    
                    evolve_gcn = self.__getattr__('evolve_gcn_final')
                    if cfg.poly_cls_branch:
                        iter_py_cls_lst = []
                        py_cls, py, final_feat = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['ind'], cfg.poly_cls_branch)
                        softmax_py_cls = F.softmax(py_cls, dim=-1)
                        iter_py_cls_lst.append(softmax_py_cls) 
                        pys.append(py/snake_config.ro)
                    else:
                        py, final_feat = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['ind'])
                        pys.append(py/snake_config.ro)

                elif cfg.csm_version == 'csm_v1':
                    py = evolve['i_it_py']
                    c_py = evolve['c_it_py']
                    pys = []
                    for i in range(self.iter):
                        evolve_gcn = self.__getattr__('evolve_gcn'+str(i))
                        py, final_feat = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['ind'])
                        py = py / snake_config.ro
                        c_py = snake_gcn_utils.img_poly_to_can_poly(py)
                        pys.append(py)
                    
                    final_loc_evolve = self.__getattr__('final_loc_evolve')
                    if cfg.poly_cls_branch:
                        final_cls_net = self.__getattr__('final_cls_net')
                    py_cls, py, final_feat = self.loc_cls_head(cnn_feature, final_loc_evolve, final_cls_net, py, c_py, init['ind'])
                    pys.append(py/snake_config.ro)
                    iter_py_cls_lst = [py_cls]

                else:
                    if cfg.poly_cls_branch:
                        iter_py_cls_lst = []
                    py, final_feat = self.evolve_poly(self.evolve_gcn, cnn_feature, evolve['i_it_py'], evolve['c_it_py'], init['ind'])
                    pys = [py / snake_config.ro]
                    for i in range(self.iter):
                        py = py / snake_config.ro
                        c_py = snake_gcn_utils.img_poly_to_can_poly(py)
                        evolve_gcn = self.__getattr__('evolve_gcn'+str(i))

                        if (i == self.iter - 1) and cfg.poly_cls_branch:
                            py_cls, py, final_feat = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['ind'], cfg.poly_cls_branch)
                            softmax_py_cls = F.softmax(py_cls, dim=-1)
                            iter_py_cls_lst.append(softmax_py_cls) 
                        else:
                            py, final_feat = self.evolve_poly(evolve_gcn, cnn_feature, py, c_py, init['ind'])
                        pys.append(py / snake_config.ro)

                if 0:
                    import matplotlib.pyplot as plt 
                    import numpy as np 
                    I = np.zeros((128,128))
                    plt.imshow(I)
                    x = evolve['i_it_py'].cpu().numpy()
                    y = [py.cpu().numpy() for py in pys]

                    for k in range(len(x)):
                        xk = x[k]
                        y0 = y[0][k]
                        y1 = y[1][k]
                        y2 = y[2][k]
                        plt.plot(xk[:,0], xk[:,1],'g')
                        plt.plot(xk[0,0], xk[0,1], 'go')
                        plt.plot(xk[1,0], xk[1,1], 'g*')
                        plt.plot(xk[2,0], xk[2,1], 'gs')

                        # plt.plot(y0[:,0], y0[:,1], 'r')
                        # plt.plot(y0[0,0], y0[0,1], 'ro')
                        # plt.plot(y0[10,0], y0[10,1], 'r*')

                        # plt.plot(y1[:,0],  y1[:,1], 'y')
                        # plt.plot(y1[0,0],  y1[0,1], 'yo')
                        # plt.plot(y1[10,0], y1[10,1], 'y*')

                        
                        # plt.plot(y2[:,0], y2[:,1], 'w')
                        # plt.plot(y2[0,0], y2[0,1], 'wo')
                        # plt.plot(y2[10,0], y2[10,1], 'w*')

                    plt.show()

                ret.update({'py': pys})
                ret.update({'final_feat': final_feat})
                if cfg.poly_cls_branch:
                    ret.update({'py_cls': iter_py_cls_lst})
    
        return output

