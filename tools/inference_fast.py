import os,time,cv2,glob
import json,tqdm
import pyclipper
import numpy as np
import torch
import torch.utils.data as data
import matplotlib.pyplot as  plt
import pycocotools.mask as maskUtils

from shapely.geometry import *
from lib.utils.snake import snake_config,snake_eval_utils,snake_poly_utils
from lib.utils import data_utils
from lib.config import cfg
from lib.networks import make_network
from lib.utils.net_utils import load_network
from lib.visualizers import make_visualizer
from tools.load_save_info import load_ctw_gt_label,load_tot_gt_label, load_msra_gt_label, \
                                 saving_mot_det_results, saving_det_results
from tools.vis_info import plot_poly, vis_dets_gts

from lib.cocoapi.PythonAPI.pycocotools.mask import frPoly, iou, rle_nms

DEBUG = True
class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        if os.path.isdir(cfg.demo_path):
            self.imgs = sorted(glob.glob(os.path.join(cfg.demo_path, '*')))
            #self.imgs = [os.path.join(cfg.demo_path, '1473.jpg')]
        elif os.path.exists(cfg.demo_path):
            self.imgs = [cfg.demo_path]
        else:
            raise Exception('NO SUCH FILE')

    def normalize_image(self, inp):
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - snake_config.mean) / snake_config.std
        inp = inp.transpose(2, 0, 1)
        return inp

    def resize(self, im, target_size, max_size, stride=0, interpolation = cv2.INTER_LINEAR):
        """
        only resize input image to target size and return scale
        :param im: BGR image input by opencv
        :param target_size: one dimensional size (the short side)
        :param max_size: one dimensional max size (the long side)
        :param stride: if given, pad the image to designated stride
        :param interpolation: if given, using given interpolation method to resize image
        :return:
        """
        im_shape = im.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        im_scale = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=interpolation)

        if stride == 0:
            return im, im_scale
        else:
            # pad to product of stride
            im_height = int(np.ceil(im.shape[0] / float(stride)) * stride)
            im_width  = int(np.ceil(im.shape[1] / float(stride)) * stride)
            im_channel = im.shape[2]
            padded_im = np.zeros((im_height, im_width, im_channel))
            padded_im[:im.shape[0], :im.shape[1], :] = im
            return padded_im, im_scale

    def __getitem__(self, index):
        img_path = self.imgs[index]
        img_name = os.path.basename(img_path)
        
        org_img = cv2.imread(img_path)
        
        if not cfg.test.target_scale:
            img = org_img.copy()
            rz_ratio = 1
        else:
            img, rz_ratio = self.resize(org_img, cfg.test.target_scale[0], cfg.test.target_scale[1])

        width, height = img.shape[1], img.shape[0]
        center = np.array([width // 2, height // 2])
        scale = np.array([width, height])
        x = 32
        input_w = (int(width / 1.) | (x - 1)) + 1
        input_h = (int(height / 1.) | (x - 1)) + 1

        trans_input = data_utils.get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)

        inp = self.normalize_image(inp)
        ret = {'inp': inp}
        meta = {'center': center, 'scale': scale, 'test': '', 'ann': ''}
        ret.update({'meta': meta})
        ret.update({'org_img': org_img})
        ret.update({'rz_img': img})
        ret.update({'rz_ratio': rz_ratio})
        ret.update({'image_name': img_name})
        return ret
    def __len__(self):
        return len(self.imgs)

def poly_rle_nms(polys, scores, img_size, nms_thresh=0.3):
    assert len(scores) ==  len(polys), 'poly.num != scores.num'
    _h, _w = img_size[0], img_size[1]
    rles = snake_eval_utils.coco_poly_to_rle(polys, _h, _w)
    iscrowd = [int(0) for k in range(len(polys))]
    ious = maskUtils.iou(rles, rles, iscrowd)
    order = scores.argsort()[::-1]
    keep = []
    while order.size>0:
        i = order[0]
        keep.append(i)
        ovr = ious[i][order[1:]].astype(np.float32)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds+1]
    return keep

def poly_rle_nms_fast(polys, scores, img_size, nms_thresh=0.3):
    _h, _w = img_size[0], img_size[1]
    polys = [ply.flatten() for ply in polys]
    rlePolys = frPoly(polys, _h, _w)
    mask_keeps = rle_nms(rlePolys, nms_thresh)
    keep_inds = np.where(mask_keeps != 0)[0]
    return keep_inds
    
def remove_close_points(points):
    next_points = np.concatenate((points[1:,:], points[0][None]))
    diff = np.min(np.abs(points - next_points), axis=-1)
    idx = diff > 1e-5
    new_points = points[idx]
    return new_points

def sorting_det_results(dets, ex_pts, polygons, contour_feat=None, poly_scores=None, sorting_flag='ct'):
    if sorting_flag == 'ct':
        scores = dets[:,4]
    else:
        scores = poly_scores
    scores_argmax_ids = np.argsort(-scores)
    sorted_dets = dets[scores_argmax_ids]
    sorted_polys = [polygons[i] for i in scores_argmax_ids]
    sorted_ex_pts = ex_pts[scores_argmax_ids]
    if poly_scores is None:
        return sorted_dets, sorted_ex_pts, sorted_polys
    else:
        sorted_poly_scores = poly_scores[scores_argmax_ids]
        sorted_contour_feat = contour_feat[scores_argmax_ids]
        return sorted_dets, sorted_ex_pts, sorted_polys, sorted_contour_feat, sorted_poly_scores


def inference():
    network = make_network(cfg).cuda()
    load_network(network, cfg.model_dir, resume=cfg.resume, epoch=cfg.test.epoch)
    network.eval()

    with open(os.path.join(cfg.results_dir,'cfg.json'),'w') as fid:
        json.dump(cfg,fid)

    dataset = Dataset()
    visualizer = make_visualizer(cfg)
    infer_time_lst = []
    for batch in tqdm.tqdm(dataset):
        batch['inp'] = torch.FloatTensor(batch['inp'])[None].cuda()
        net_time_s = time.time()
        with torch.no_grad():
            output = network(batch['inp'], batch)
        net_used_time = time.time()-net_time_s

        org_img = batch['org_img']
        rz_img = batch['rz_img']
        rz_ratio = batch['rz_ratio']
        img_name = batch['image_name']
        center = batch['meta']['center']
        scale = batch['meta']['scale']
        h, w = batch['inp'].size(2), batch['inp'].size(3)
        
        detections = output['detection'].detach().cpu().numpy()
        detections[:,:4] = detections[:, :4] * snake_config.down_ratio
        bboxes = detections[:, :4]
        scores = detections[:, 4]
        labels = detections[:, 5].astype(int)
        ex_pts = output['ex'].detach().cpu().numpy()
        ex_pts = ex_pts * snake_config.down_ratio
        #pys = output['py'][-1].detach().cpu().numpy() * snake_config.down_ratio
        iter_ply_output_lst = [x.detach().cpu().numpy()* snake_config.down_ratio for x in output['py']]
        pys = iter_ply_output_lst[-1]

        final_contour_feat = output['final_feat'].detach().cpu().numpy()
        if cfg.poly_cls_branch:
            pys_cls = output['py_cls'][-1].detach().cpu().numpy()
            text_poly_scores = pys_cls[:, 1]
            rem_ids = np.where(text_poly_scores > cfg.poly_conf_thresh)[0]
            detections = detections[rem_ids]
            pys = pys[rem_ids]
            text_poly_scores = text_poly_scores[rem_ids]
            ex_pts = ex_pts[rem_ids]
            final_contour_feat = final_contour_feat[rem_ids]

        if len(pys) == 0:
            all_boundaries, poly_scores = [], []
        else:
            trans_output_inv = data_utils.get_affine_transform(center, scale, 0, [w, h], inv=1)
            all_boundaries   = [data_utils.affine_transform(py_, trans_output_inv) for py_ in pys]
            bboxes_tmp = [data_utils.affine_transform(det[:4].reshape(-1,2), trans_output_inv).flatten() for det in detections]
            ex_pts_tmp = [data_utils.affine_transform(ep, trans_output_inv) for ep in ex_pts]
            detections = np.hstack((np.array(bboxes_tmp), detections[:,4:]))
            ex_pts = np.array(ex_pts_tmp)

            pp_time_s = time.time()
            #sorting detections by scores
            if cfg.poly_cls_branch:
                detections, ex_points, all_boundaries, final_contour_feat, poly_scores \
                  = sorting_det_results(detections, ex_pts, all_boundaries, final_contour_feat, text_poly_scores)
            else:
                detections, ex_points, all_boundaries = sorting_det_results(detections, ex_pts, all_boundaries)
           
            if cfg.rle_nms:
                tmp_polys = all_boundaries.copy()
                #all_boundaries, rem_inds = snake_poly_utils.poly_nms(tmp_polys)
                rem_inds = poly_rle_nms(tmp_polys, detections[:,-1], (h, w), nms_thresh=0.3)
                all_boundaries = [all_boundaries[idx] for idx in rem_inds]
            else:
                #nms
                all_boundaries, rem_inds = snake_poly_utils.poly_nms(all_boundaries)
            detections = detections[rem_inds]
            ex_points = ex_points[rem_inds]
            final_contour_feat = final_contour_feat[rem_inds]
            if cfg.poly_cls_branch:
                poly_scores = poly_scores[rem_inds]
            pp_used_time = time.time() - pp_time_s
            infer_time_lst.append([net_used_time, pp_used_time])
          
            if len(all_boundaries) != 0:
                detections[:,:4] /= rz_ratio
                ex_points /= rz_ratio
                all_boundaries = [poly/rz_ratio for poly in all_boundaries]
            
        #--------------------------------saving results-------------------------------#
        det_file = os.path.join(cfg.det_dir, (img_name[:-4]+'.txt'))
        saving_det_results(det_file, all_boundaries, img=org_img)
        
       
       