import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.config import cfg

DEBUG = False
class StandardConv(nn.Module):
    def __init__(self, state_dim, out_state_dim=None, n_adj=4, dilation=1):
        super(StandardConv, self).__init__()
        self.n_adj = n_adj
        self.dilation = dilation
        out_state_dim = state_dim if out_state_dim is None else out_state_dim
        self.fc = nn.Conv1d(state_dim, out_state_dim, kernel_size=self.n_adj*2+1, dilation=self.dilation, padding=self.n_adj*self.dilation)

    def forward(self, _input, adj=None):
        if self.n_adj != 0:
            return self.fc(_input)

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
    'dgrid': DilatedCircConv,
    'sgrid': StandardConv
}
class PSPModule(nn.Module):
    def __init__(self, sizes=(1, 3, 6, 8), dimension=1):
        super(PSPModule, self).__init__()
        self.stages = nn.ModuleList([self._make_stage(size, dimension) for size in sizes])

    def _make_stage(self, size, dimension=2):
        if dimension == 1:
            prior = nn.AdaptiveAvgPool1d(output_size=size)
        elif dimension == 2:
            prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        return prior

    def forward(self, feats):
        n, c, _ = feats.size()
        priors = [stage(feats) for stage in self.stages]
        center = torch.cat(priors, -1)
        return center


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

class BasicBlock_v4(nn.Module):
    def __init__(self, state_dim, out_state_dim, conv_type, n_adj=4, dilation=1, psp_size=(1,16)):
        super(BasicBlock_v4, self).__init__()
        self.out_state_dim = out_state_dim
        self.psp = PSPModule(psp_size)
        self.f_key = nn.Sequential(
            _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(out_state_dim),
            #nn.GroupNorm(4, out_state_dim),
        )
        self.f_query = self.f_key
        self.f_value = nn.Sequential(
                        _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation),
                        nn.ReLU(inplace=True),
                        nn.BatchNorm1d(out_state_dim),
                        #nn.GroupNorm(4, out_state_dim),
                      )
        self.W = _conv_factory[conv_type](state_dim, out_state_dim, n_adj, dilation)
        #self.fusion = nn.Conv1d(state_dim, out_state_dim, 1)
        #nn.init.constant_(self.W.weight, 0)
        #nn.init.constant_(self.W.bias, 0)

    def forward(self, x, adj=None):
        query = self.f_query(x)
        key = self.f_key(x)
        key = self.psp(key)  
        sim_map = torch.matmul(query.permute(0,2,1), key)
        sim_map = (self.out_state_dim ** -.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        value = self.psp(self.f_value(x))
        context = torch.matmul(sim_map, value.permute(0,2,1))
        context = context.permute(0, 2, 1).contiguous()
        return context

class Snake(nn.Module):
    def __init__(self, state_dim, feature_dim, conv_type='dgrid', snake_type='evolve'):
        super(Snake, self).__init__()
        if 'CIA_v4' in cfg.snake_net:
            self.head = BasicBlock(feature_dim, state_dim, conv_type)
            dilation = [1, 1, 1, 2, 2, 4, 4]
            self.res_layer_num = len(dilation)
            for i in range(self.res_layer_num):
                conv = BasicBlock_v4(state_dim, state_dim, conv_type, n_adj=4, dilation=dilation[i],psp_size=cfg.psp_size)
                self.__setattr__('res'+str(i), conv)
            fusion_state_dim = 256
            self.fusion = nn.Conv1d(state_dim * (self.res_layer_num + 1), fusion_state_dim, 1)
        
        if cfg.snake_net == 'CIA_v4':
            predhead_fdim = state_dim * (self.res_layer_num + 1) + fusion_state_dim
        elif cfg.snake_net == 'no_cia':
            predhead_fdim = feature_dim
        else:
            raise ValueError('Not supported snake network:', cfg.snake_net)
        
        ## Predition head network
        self.snake_type = snake_type
        if cfg.poly_cls_branch and (self.snake_type=='evolve_final'):
            self.poly_cls_proj = nn.Conv1d(predhead_fdim, 256, 1)
            self.linear1 = nn.Linear(256*2, 512, bias=False)
            self.bn1 = nn.BatchNorm1d(512)
            self.dp1 = nn.Dropout(p=0.5)
            self.linear2 = nn.Linear(512, 256)
            self.bn2 = nn.BatchNorm1d(256)
            self.dp2 = nn.Dropout(p=0.5)
            self.linear3 = nn.Linear(256, 2)
            
        self.prediction = nn.Sequential(
            nn.Conv1d(predhead_fdim, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 64, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(64, 2, 1)
        )

    def forward(self, x, adj):
        pts_feats = x.clone()
        if 'CIA_v4' in cfg.snake_net:
            states = []
            x = self.head(x, adj)
            states.append(x)
            for i in range(self.res_layer_num):
                x = self.__getattr__('res'+str(i))(x, adj) + x
                states.append(x)
            state = torch.cat(states, dim=1)
            global_state = torch.max(self.fusion(state), dim=2, keepdim=True)[0]
            global_state = global_state.expand(global_state.size(0), global_state.size(1), state.size(2))
            state = torch.cat([global_state, state], dim=1)
        elif 'no_cia' in cfg.snake_net:
            state = x.clone()
        else:
            raise ValueError('Not supported snake network:', cfg.snake_net)

        ## Predition Head
        x = self.prediction(state)
        if cfg.poly_cls_branch and (self.snake_type=='evolve_final'):
            if cfg.dyn_poly_cls:
                state = self.creating_dynamic_feat(pts_feats)
            if cfg.poly_cls_branch_type == 'v1':
                y = self.poly_cls_proj(state)
                instance_num, _, _ = y.shape
                y = F.adaptive_max_pool1d(y, 1).view(instance_num, -1)
                y = F.relu(self.bn1(self.linear1(y)))
                y = self.dp1(y)
                y = F.relu(self.bn2(self.linear2(y)))
                y = self.dp2(y)
                y = self.linear3(y)
            else:
                y = self.poly_cls_proj(state)
                instance_num, _, _ = y.shape
                y_avg = F.adaptive_avg_pool1d(y, 1).view(instance_num, -1)
                y_max = F.adaptive_max_pool1d(y, 1).view(instance_num, -1)
                y = torch.cat((y_avg, y_max), 1)
                y = F.leaky_relu(self.bn1(self.linear1(y)), negative_slope=0.2)
                y = self.dp1(y)
                y = F.leaky_relu(self.bn2(self.linear2(y)), negative_slope=0.2)
                y = self.dp2(y)
                y = self.linear3(y)
            return x, y
        return x
