import torch
import os
import math
from torch import nn
import numpy as np
import torch.nn.functional
from collections import OrderedDict
from termcolor import colored
from sklearn.utils.extmath import cartesian

DEBUG = False
def sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def _neg_loss(pred, gt):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
            pred (batch x c x h x w)
            gt_regr (batch x c x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()


    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


class FocalLoss(nn.Module):
    '''nn.Module warpper for focal loss'''
    def __init__(self):
        super(FocalLoss, self).__init__()
        self.neg_loss = _neg_loss

    def forward(self, out, target):
        return self.neg_loss(out, target)


def smooth_l1_loss(vertex_pred, vertex_targets, vertex_weights, sigma=1.0, normalize=True, reduce=True):
    """
    :param vertex_pred:     [b, vn*2, h, w]
    :param vertex_targets:  [b, vn*2, h, w]
    :param vertex_weights:  [b, 1, h, w]
    :param sigma:
    :param normalize:
    :param reduce:
    :return:
    """
    b, ver_dim, _, _ = vertex_pred.shape
    sigma_2 = sigma ** 2
    vertex_diff = vertex_pred - vertex_targets
    diff = vertex_weights * vertex_diff
    abs_diff = torch.abs(diff)
    smoothL1_sign = (abs_diff < 1. / sigma_2).detach().float()
    in_loss = torch.pow(diff, 2) * (sigma_2 / 2.) * smoothL1_sign \
              + (abs_diff - (0.5 / sigma_2)) * (1. - smoothL1_sign)

    if normalize:
        in_loss = torch.sum(in_loss.view(b, -1), 1) / (ver_dim * torch.sum(vertex_weights.view(b, -1), 1) + 1e-3)

    if reduce:
        in_loss = torch.mean(in_loss)

    return in_loss


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super(SmoothL1Loss, self).__init__()
        self.smooth_l1_loss = smooth_l1_loss

    def forward(self, preds, targets, weights, sigma=1.0, normalize=True, reduce=True):
        return self.smooth_l1_loss(preds, targets, weights, sigma, normalize, reduce)


class AELoss(nn.Module):
    def __init__(self):
        super(AELoss, self).__init__()

    def forward(self, ae, ind, ind_mask):
        """
        ae: [b, 1, h, w]
        ind: [b, max_objs, max_parts]
        ind_mask: [b, max_objs, max_parts]
        obj_mask: [b, max_objs]
        """
        # first index
        b, _, h, w = ae.shape
        b, max_objs, max_parts = ind.shape
        obj_mask = torch.sum(ind_mask, dim=2) != 0

        ae = ae.view(b, h * w, 1)
        seed_ind = ind.view(b, max_objs * max_parts, 1)
        tag = ae.gather(1, seed_ind).view(b, max_objs, max_parts)

        # compute the mean
        tag_mean = tag * ind_mask
        tag_mean = tag_mean.sum(2) / (ind_mask.sum(2) + 1e-4)

        # pull ae of the same object to their mean
        pull_dist = (tag - tag_mean.unsqueeze(2)).pow(2) * ind_mask
        obj_num = obj_mask.sum(dim=1).float()
        pull = (pull_dist.sum(dim=(1, 2)) / (obj_num + 1e-4)).sum()
        pull /= b

        # push away the mean of different objects
        push_dist = torch.abs(tag_mean.unsqueeze(1) - tag_mean.unsqueeze(2))
        push_dist = 1 - push_dist
        push_dist = nn.functional.relu(push_dist, inplace=True)
        obj_mask = (obj_mask.unsqueeze(1) + obj_mask.unsqueeze(2)) == 2
        push_dist = push_dist * obj_mask.float()
        push = ((push_dist.sum(dim=(1, 2)) - obj_num) / (obj_num * (obj_num - 1) + 1e-4)).sum()
        push /= b
        return pull, push


class PolyMatchingLoss(nn.Module):
    def __init__(self, pnum):
        super(PolyMatchingLoss, self).__init__()

        self.pnum = pnum
        batch_size = 1
        pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)
        for b in range(batch_size):
            for i in range(pnum):
                pidx = (np.arange(pnum) + i) % pnum
                pidxall[b, i] = pidx

        device = torch.device('cuda')
        pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1))).to(device)

        self.feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), 2).detach()

    def forward(self, pred, gt, loss_type="L2"):
        pnum = self.pnum
        batch_size = pred.size()[0]
        feature_id = self.feature_id.expand(batch_size, self.feature_id.size(1), 2)
        device = torch.device('cuda')

        gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, pnum, pnum, 2)

        pred_expand = pred.unsqueeze(1)

        dis = pred_expand - gt_expand

        if loss_type == "L2":
            dis = (dis ** 2).sum(3).sqrt().sum(2)
        elif loss_type == "L1":
            dis = torch.abs(dis).sum(3).sum(2)

        min_dis, min_id = torch.min(dis, dim=1, keepdim=True)
        # print(min_id)

        # min_id = torch.from_numpy(min_id.data.cpu().numpy()).to(device)
        # min_gt_id_to_gather = min_id.unsqueeze_(2).unsqueeze_(3).long().\
        #                         expand(min_id.size(0), min_id.size(1), gt_expand.size(2), gt_expand.size(3))
        # gt_right_order = torch.gather(gt_expand, 1, min_gt_id_to_gather).view(batch_size, pnum, 2)

        return torch.mean(min_dis)

class DistanceConstraint(nn.Module):
    def __init__(self, type='log'):
        super(DistanceConstraint, self).__init__()
        self.dist_type = type

    def forward(self, output, target, ind, weight):
        """ind: [b, n]"""
        output = _tranpose_and_gather_feat(output, ind)
        weight = weight.unsqueeze(2)
        valid_output = output * weight
        valid_target = target * weight
        pred_w1 = valid_output[...,0:1]
        pred_h1 = valid_output[...,1:2]
        pred_w2 = valid_output[...,2:3]
        pred_h2 = valid_output[...,3:4]

        x1 = torch.min(pred_w1, pred_w2) / (torch.max(pred_w1, pred_w2) + 1e-6)
        x2 = torch.min(pred_h1, pred_h2) / (torch.max(pred_h1, pred_h2) + 1e-6)
        dist_constraint_loss = -torch.log(x1 + x2)
        #print('dist_constraint_loss.size:', dist_constraint_loss.size())
        
        dist_constraint_loss = torch.sum(dist_constraint_loss)/(torch.sum(weight) + 1e-6)
        return dist_constraint_loss

class RegionIOU(nn.Module):
    def __init__(self, type='log'):
        super(RegionIOU, self).__init__()
        if type == 'log':
            self.normalize_factor = True

    def forward(self, output, target, ind, weight):
        """ind: [b, n]"""
        output = _tranpose_and_gather_feat(output, ind)
        weight = weight.unsqueeze(2)

        if 0:
            print('---------------region iou loss forward------------------')
            print('output.size:', output.size())
            print('weight.size:', weight.size())
            print('target.size:', target.size())
            print('weight:', weight)
        valid_output = output * weight
        valid_target = target * weight
        gt_w1 = valid_target[...,0:1]
        gt_h1 = valid_target[...,1:2]
        gt_w2 = valid_target[...,2:3]
        gt_h2 = valid_target[...,3:4]
        pred_w1 = valid_output[...,0:1]
        pred_h1 = valid_output[...,1:2]
        pred_w2 = valid_output[...,2:3]
        pred_h2 = valid_output[...,3:4]
        gt_area = (gt_w1 + gt_w2) * (gt_h1 + gt_h2)
        pred_area = (pred_w1 + pred_w2) * (pred_h1 + pred_h2)
        union_w = torch.min(gt_w1, pred_w1) + torch.min(gt_w2, pred_w2)
        union_h = torch.min(gt_h1, pred_h1) + torch.min(gt_h2, pred_h2)
        intersect_area = union_w * union_h
        union_area = gt_area + pred_area - intersect_area
        iou = -torch.log((intersect_area + 1.0)/(union_area + 1.0))
        avg_iou = torch.sum(iou)/(torch.sum(weight)+1e-6)
        return avg_iou
class AveragedHausdorffLoss(nn.Module):
    def __init__(self):
        super(AveragedHausdorffLoss, self).__init__()
    def cdist(self, x, y):
        """
        Compute distance between each pair of the two collections of inputs.
        :param x: Nxd Tensor
        :param y: Mxd Tensor
        :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
              i.e. dist[i,j] = ||x[i,:]-y[j,:]||

        """
        differences = x.unsqueeze(1) - y.unsqueeze(0)
        distances = torch.sum(differences**2, -1).sqrt()
        return distances

    def forward(self, set1, set2):
        """
        Compute the Averaged Hausdorff Distance function
        between two unordered sets of points (the function is symmetric).
        Batches are not supported, so squeeze your inputs first!
        :param set1: Tensor where each row is an N-dimensional point.
        :param set2: Tensor where each row is an N-dimensional point.
        :return: The Averaged Hausdorff Distance between set1 and set2.
        """

        assert set1.size()[0] == set2.size()[0], 'Batch size is inconsistent!'
        assert set1.size()[1] == set2.size()[1], 'The points in both sets must have the same number of dimensions, got %s and %s.'% (set2.size()[1], set2.size()[1])
        
        instance_num = set1.size()[0]
        ahd = 0
        for k in range(instance_num):
            inst1_pts = set1[k]
            inst2_pts = set2[k]

            d2_matrix = self.cdist(inst1_pts, inst2_pts)

            # Modified Chamfer Loss
            term_1 = torch.mean(torch.min(d2_matrix, 1)[0])
            term_2 = torch.mean(torch.min(d2_matrix, 0)[0])

            res = term_1 + term_2
            ahd += res
        
        if instance_num != 0:
            ahd = ahd / instance_num
        return ahd

class WeightedHausdorffDistance(nn.Module):
    def __init__(self,
                 resized_height, 
                 resized_width,
                 alpha = -9,
                 return_2_terms=False,
                 device=torch.device('cuda')):
        """
        :param resized_height: Number of rows in the image.
        :param resized_width: Number of columns in the image.
        :param p: Exponent in the generalized mean. -inf makes it the minimum.
        :param return_2_terms: Whether to return the 2 terms
                               of the WHD instead of their sum.
                               Default: False.
        :param device: Device where all Tensors will reside.
        """
        super(WeightedHausdorffDistance, self).__init__()

        # Prepare all possible (row, col) locations in the image
        self.height, self.width = int(resized_height), int(resized_width)
        self.resized_size = torch.tensor([resized_height,
                                          resized_width],
                                         dtype=torch.get_default_dtype(),
                                         device=device)

        self.max_dist = math.sqrt(resized_height**2 + resized_width**2)
        self.n_pixels = resized_height * resized_width
        self.all_img_locations = torch.from_numpy(cartesian([np.arange(resized_height),
                                                             np.arange(resized_width)]))
        # Convert to appropiate type
        self.all_img_locations = self.all_img_locations.to(device=device,
                                                           dtype=torch.get_default_dtype())

        self.return_2_terms = return_2_terms
        self.alpha = alpha

    def _assert_no_grad(self, variables):
        for var in variables:
            assert not var.requires_grad, \
                "nn criterions don't compute the gradient w.r.t. targets - please " \
                "mark these variables as volatile or not requiring gradients"
    def cdist(self, x, y):
        """
        Compute distance between each pair of the two collections of inputs.
        :param x: Nxd Tensor
        :param y: Mxd Tensor
        :res: NxM matrix where dist[i,j] is the norm between x[i,:] and y[j,:],
              i.e. dist[i,j] = ||x[i,:]-y[j,:]||
        """
        differences = x.unsqueeze(1) - y.unsqueeze(0)
        distances = torch.sum(differences**2, -1).sqrt()
        return distances

    def generaliz_mean(self, tensor, dim, alpha=-9, keepdim=False):
        # """
        # Computes the softmin along some axes.
        # Softmin is the same as -softmax(-x), i.e,
        # softmin(x) = -log(sum_i(exp(-x_i)))

        # The smoothness of the operator is controlled with k:
        # softmin(x) = -log(sum_i(exp(-k*x_i)))/k

        # :param input: Tensor of any dimension.
        # :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
        # :param keepdim: (bool) Whether the output tensor has dim retained or not.
        # :param k: (float>0) How similar softmin is to min (the lower the more smooth).
        # """
        # return -torch.log(torch.sum(torch.exp(-k*input), dim, keepdim))/k
        """
        The generalized mean. It corresponds to the minimum when p = -inf.
        https://en.wikipedia.org/wiki/Generalized_mean
        :param tensor: Tensor of any dimension.
        :param dim: (int or tuple of ints) The dimension or dimensions to reduce.
        :param keepdim: (bool) Whether the output tensor has dim retained or not.
        :param p: (float<0).
        """
        assert alpha < 0, 'alpha should be less than 0.'
        res= torch.mean((tensor + 1e-6)**alpha, dim, keepdim=keepdim)**(1./alpha)
        return res
        
    def forward(self, prob_map, gt_map):
        """
        Compute the Weighted Hausdorff Distance function
        between the estimated probability map and ground truth points.
        The output is the WHD averaged through all the batch.

        :param prob_map: (B x H x W) Tensor of the probability map of the estimation.
                         B is batch size, H is height and W is width.
                         Values must be between 0 and 1.
        :param gt: List of Tensors of the Ground Truth points.
                   Must be of size B as in prob_map.
                   Each element in the list must be a 2D Tensor,
                   where each row is the (y, x), i.e, (row, col) of a GT point.
        :param orig_sizes: Bx2 Tensor containing the size
                           of the original images.
                           B is batch size.
                           The size must be in (height, width) format.
        :param orig_widths: List of the original widths for each image
                            in the batch.
        :return: Single-scalar Tensor with the Weighted Hausdorff Distance.
                 If self.return_2_terms=True, then return a tuple containing
                 the two terms of the Weighted Hausdorff Distance.
        """

        #self._assert_no_grad(gt)
        if DEBUG:
            print('prob_map.shape:', prob_map.shape)
            print('gt_map.shape:', gt_map.shape)


        assert prob_map.size()[2:4] == (self.height, self.width), \
            'You must configure the WeightedHausdorffDistance with the height and width of the ' \
            'probability map that you are using, got a probability map of size %s'\
            % str(prob_map.size())

        batch_size = prob_map.shape[0]
        assert batch_size == gt_map.shape[0],'Batch num for ouput != batch num of gt'
        terms_1, terms_2 = [], []
        for b in range(batch_size):
            # One by one
            prob_map_b = prob_map[b, 0, ...]
            gt_b = gt_map[b, 0, ...]
            gt_pts = (gt_b == 1).nonzero()
            n_gt_pts = gt_pts.size()[0]
            if DEBUG:
                print('n_gt_pts:', n_gt_pts)
                print('gt_pts:', gt_pts)
                import matplotlib.pyplot as plt 
                a = gt_b.cpu().numpy()
                b = gt_pts.cpu().numpy()
                plt.imshow(a)
                for k in range(len(b)):
                    plt.plot(b[k][1], b[k][0],'g+')
                plt.savefig('gt_pt_map.png')
                plt.close()
            # Corner case: no GT points
            if n_gt_pts == 0:
                terms_1.append(torch.tensor([0],
                                            dtype=torch.get_default_dtype()))
                terms_2.append(torch.tensor([self.max_dist],
                                            dtype=torch.get_default_dtype()))
                continue
            
            if DEBUG:
                print('loc.size:', self.all_img_locations.size())
                print('gt_pts.size:', gt_pts.size())

            # Pairwise distances between all possible locations and the GTed locations
            normalized_x =  self.all_img_locations.float()
            normalized_y =  gt_pts.float()

            d_matrix = self.cdist(normalized_x, normalized_y)

            if DEBUG:
                print('d_matrix.shape:', d_matrix.size())
            # Reshape probability map as a long column vector,
            # and prepare it for multiplication
            p = prob_map_b.view(prob_map_b.nelement())

            n_est_pts = p.sum()
            p_replicated = p.view(-1, 1).repeat(1, n_gt_pts)

            # Weighted Hausdorff Distance
            term_1 = (1 / (n_est_pts + 1e-6)) * torch.sum(p * torch.min(d_matrix, 1)[0])
            weighted_d_matrix = (1 - p_replicated) * self.max_dist + p_replicated * d_matrix
            minn = self.generaliz_mean(weighted_d_matrix, alpha=self.alpha, dim=0, keepdim=False)
            term_2 = torch.mean(minn)
            if DEBUG:
                print('minn:', minn)
            terms_1.append(term_1)
            terms_2.append(term_2)

        terms_1 = torch.stack(terms_1)
        terms_2 = torch.stack(terms_2)

        if self.return_2_terms:
            res = terms_1.mean(), terms_2.mean()
        else:
            res = terms_1.mean() + terms_2.mean()
        
        return res

class AttentionLoss(nn.Module):
    def __init__(self, beta=4, gamma=0.5):
        super(AttentionLoss, self).__init__()

        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, gt):
        num_pos = torch.sum(gt)
        num_neg = torch.sum(1 - gt)
        alpha = num_neg / (num_pos + num_neg)
        edge_beta = torch.pow(self.beta, torch.pow(1 - pred, self.gamma))
        bg_beta = torch.pow(self.beta, torch.pow(pred, self.gamma))

        loss = 0
        loss = loss - alpha * edge_beta * torch.log(pred) * gt
        loss = loss - (1 - alpha) * bg_beta * torch.log(1 - pred) * (1 - gt)
        return torch.mean(loss)


def _gather_feat(feat, ind, mask=None):
    dim = feat.size(2)
    ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _tranpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class Ind2dRegL1Loss(nn.Module):
    def __init__(self, type='l1'):
        super(Ind2dRegL1Loss, self).__init__()
        if type == 'l1':
            self.loss = torch.nn.functional.l1_loss
        elif type == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss

    def forward(self, output, target, ind, ind_mask):
        """ind: [b, max_objs, max_parts]"""
        b, max_objs, max_parts = ind.shape
        ind = ind.view(b, max_objs * max_parts)
        pred = _tranpose_and_gather_feat(output, ind).view(b, max_objs, max_parts, output.size(1))
        mask = ind_mask.unsqueeze(3).expand_as(pred)
        loss = self.loss(pred * mask, target * mask, reduction='sum')
        loss = loss / (mask.sum() + 1e-4)
        return loss


class IndL1Loss1d(nn.Module):
    def __init__(self, type='l1'):
        super(IndL1Loss1d, self).__init__()
        if type == 'l1':
            self.loss = torch.nn.functional.l1_loss
        elif type == 'smooth_l1':
            self.loss = torch.nn.functional.smooth_l1_loss

    def forward(self, output, target, ind, weight):
        """ind: [b, n]"""
        output = _tranpose_and_gather_feat(output, ind)
        weight = weight.unsqueeze(2)
        loss = self.loss(output * weight, target * weight, reduction='sum')
        loss = loss / (weight.sum() * output.size(2) + 1e-4)
        return loss


class GeoCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(GeoCrossEntropyLoss, self).__init__()

    def forward(self, output, target, poly):
        output = torch.nn.functional.softmax(output, dim=1)
        output = torch.log(torch.clamp(output, min=1e-4))
        poly = poly.view(poly.size(0), 4, poly.size(1) // 4, 2)
        target = target[..., None, None].expand(poly.size(0), poly.size(1), 1, poly.size(3))
        target_poly = torch.gather(poly, 2, target)
        sigma = (poly[:, :, 0] - poly[:, :, 1]).pow(2).sum(2, keepdim=True)
        kernel = torch.exp(-(poly - target_poly).pow(2).sum(3) / (sigma / 3))
        loss = -(output * kernel.transpose(2, 1)).sum(1).mean()
        return loss


def load_model(net, optim, scheduler, recorder, model_dir, resume=True, epoch=-1):
    if not resume:
        os.system('rm -rf {}'.format(model_dir))
        return 0
    
    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
  
    net.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    scheduler.load_state_dict(pretrained_model['scheduler'])
    recorder.load_state_dict(pretrained_model['recorder'])
    return pretrained_model['epoch'] + 1


def save_model(net, optim, scheduler, recorder, epoch, model_dir):
    os.system('mkdir -p {}'.format(model_dir))
    torch.save({
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'scheduler': scheduler.state_dict(),
        'recorder': recorder.state_dict(),
        'epoch': epoch
    }, os.path.join(model_dir, '{}.pth'.format(epoch)))

    # remove previous pretrained model if the number of models is too big
    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) <= 200:
        return
    os.system('rm {}'.format(os.path.join(model_dir, '{}.pth'.format(min(pths)))))


def load_network(net, model_dir, resume=True, epoch=-1, strict=True):
    if not resume:
        return 0

    if not os.path.exists(model_dir):
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir) if 'pth' in pth]
    if len(pths) == 0:
        print(colored('WARNING: NO MODEL LOADED !!!', 'red'))
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    print('load model: {}'.format(os.path.join(model_dir, '{}.pth'.format(pth))))
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    net.load_state_dict(pretrained_model['net'], strict=strict)
    return pretrained_model['epoch'] + 1


def remove_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(prefix):
            net_[k[len(prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def add_net_prefix(net, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        net_[prefix + k] = net[k]
    return net_


def replace_net_prefix(net, orig_prefix, prefix):
    net_ = OrderedDict()
    for k in net.keys():
        if k.startswith(orig_prefix):
            net_[prefix + k[len(orig_prefix):]] = net[k]
        else:
            net_[k] = net[k]
    return net_


def remove_net_layer(net, layers):
    keys = list(net.keys())
    for k in keys:
        for layer in layers:
            if k.startswith(layer):
                del net[k]
    return net
