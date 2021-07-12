import os
import cv2
import math
import pyclipper
import numpy as np
from lib.config import cfg
from lib.utils import data_utils
import torch.utils.data as data
from pycocotools.coco import COCO
from lib.utils.snake import snake_text_utils, snake_config, visualize_utils

DEBUG = False
class Dataset(data.Dataset):
    def __init__(self, ann_file, data_root, split):
        super(Dataset, self).__init__()

        self.data_root = data_root
        self.split = split

        self.coco = COCO(ann_file)
        self.anns = np.array(sorted(self.coco.getImgIds()))
        self.anns = self.anns[:500] if split == 'mini' else self.anns
        self.json_category_id_to_contiguous_id = {v: i for i, v in enumerate(self.coco.getCatIds())}
        
    def process_info(self, img_id):
        
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        
        anno = self.coco.loadAnns(ann_ids)
        path = os.path.join(self.data_root, self.coco.loadImgs(int(img_id))[0]['file_name'])
        
        return anno, path, img_id

    def read_original_data(self, anno, path):
        img = cv2.imread(path)
        instance_polys = [[np.array(poly).reshape(-1, 2) for poly in obj['segmentation']] for obj in anno]
        cls_ids = [self.json_category_id_to_contiguous_id[obj['category_id']] for obj in anno]
        return img, instance_polys, cls_ids

    def transform_original_data(self, instance_polys, flipped, width, trans_output, inp_out_hw):
        
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            polys = [poly.reshape(-1, 2) for poly in instance]

            if flipped:
                polys_ = []
                for poly in polys:
                    poly[:, 0] = width - np.array(poly[:, 0]) - 1
                    polys_.append(poly.copy())
                polys = polys_

            polys = snake_text_utils.transform_polys(polys, trans_output, output_h, output_w)
            instance_polys_.append(polys)
        return instance_polys_

    def get_valid_polys(self, instance_polys, inp_out_hw):
        output_h, output_w = inp_out_hw[2:]
        instance_polys_ = []
        for instance in instance_polys:
            instance = [poly for poly in instance if len(poly) >= 4]
            for poly in instance:
                poly[:, 0] = np.clip(poly[:, 0], 0, output_w - 1)
                poly[:, 1] = np.clip(poly[:, 1], 0, output_h - 1)
            polys = snake_text_utils.filter_tiny_polys(instance)
            polys = snake_text_utils.get_cw_polys(polys)
            polys = [poly[np.sort(np.unique(poly, axis=0, return_index=True)[1])] for poly in polys]
            instance_polys_.append(polys)
        return instance_polys_

    def get_extreme_points(self, instance_polys):
        
        extreme_points = []
        for instance in instance_polys:
            points = [snake_text_utils.get_extreme_points(poly) for poly in instance]
            extreme_points.append(points)

        return extreme_points


    def get_corner_points(self, instance_polys):
        def get_cn_pts(poly):
            '''
            params: points 4x2
            '''
            import cv2
            rect = cv2.minAreaRect(np.round(poly).astype(np.int32))
            corner_pts = cv2.boxPoints(rect)

            #resorting order
            center_x = np.mean(corner_pts[:,0])
            center_y = np.mean(corner_pts[:,1])
            left_ids = np.where(corner_pts[:,0]<= center_x)[0]
            left_points = corner_pts[left_ids]
            right_ids = np.where(corner_pts[:,0] > center_x)[0]
            right_points = corner_pts[right_ids]

            if len(left_points) != len(right_points):
                argmin = np.argmin(corner_pts[:,0])
                argmax = np.argmax(corner_pts[:,0])
                min1_p = corner_pts[None, argmin]
                max1_p = corner_pts[None, argmax]
                rem_ind = set(np.arange(len(corner_pts)))-set([argmin,argmax])
                rem_points = corner_pts[list(rem_ind)]
                argmin = np.argmin(rem_points[:,0])
                min2_p = rem_points[None,argmin]
                max2_p = np.delete(rem_points, argmin,axis=0)
                left_points = np.vstack((min1_p, min2_p))
                right_points = np.vstack((max1_p, max2_p))
                

            if left_points[0,1] <= left_points[1,1]:
                p1 = left_points[0]
                p4 = left_points[1]
            else:
                p1 = left_points[1]
                p4 = left_points[0]

            if right_points[0,1] <= right_points[1,1]:
                p2 = right_points[0]
                p3 = right_points[1]
            else:
                p2 = right_points[1]
                p3 = right_points[0]
            
            #pp = np.hstack((p1,p2,p3,p4)).reshape(-1, 2)
            pp = np.hstack((p1, p4, p3, p2)).reshape(-1, 2) #anti-clockwise
            
            return pp

        corner_points = []
        for instance in instance_polys:
            points = [get_cn_pts(poly) for poly in instance]
            corner_points.append(points)
        return corner_points

    def prepare_detection(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind, ct_reg_pts=None, centerness=None, rs_hm=None):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)
        x_min, y_min, x_max, y_max = box
        ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
        ct_float = ct.copy()
        ct = np.round(ct).astype(np.int32)
        
        if ct_reg_pts is not None:
            ct_reg_pts.append((ct_float  - ct).tolist())

        if cfg.dist_4d:
            xc = np.mean(poly[:, 0])
            yc = np.mean(poly[:, 1])
            w1 = xc - x_min
            w2 = x_max - xc
            h1 = yc -y_min
            h2 = y_max -yc
            h = h1 + h2
            w = w1 + w2
            wh.append([w1,h1,w2,h2])
            if cfg.centerness:
                c = 1.0
                centerness.append([c])
        else:
            h, w = y_max - y_min, x_max - x_min
            wh.append([w, h])
        
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])
        
        
        x_min, y_min = ct[0] - w / 2, ct[1] - h / 2
        x_max, y_max = ct[0] + w / 2, ct[1] + h / 2
        decode_box = [x_min, y_min, x_max, y_max]
       
        return decode_box

    def prepare_detection_(self, box, poly, ct_hm, cls_id, wh, ct_cls, ct_ind, ct_reg_pts=None):
        ct_hm = ct_hm[cls_id]
        ct_cls.append(cls_id)

        x_min, y_min, x_max, y_max = box
        box_ct = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)

        x_min_int, y_min_int = int(x_min), int(y_min)
        h_int, w_int = math.ceil(y_max - y_min_int) + 1, math.ceil(x_max - x_min_int) + 1
        max_h, max_w = ct_hm.shape[0], ct_hm.shape[1]
        h_int, w_int = min(y_min_int + h_int, max_h) - y_min_int, min(x_min_int + w_int, max_w) - x_min_int

        mask_poly = poly - np.array([x_min_int, y_min_int])
        mask_ct = box_ct - np.array([x_min_int, y_min_int])
        ct, off, xy = snake_text_utils.prepare_ct_off_mask(mask_poly, mask_ct, h_int, w_int)

        xy += np.array([x_min_int, y_min_int])
        ct += np.array([x_min_int, y_min_int])

        if cfg.dist_4d:
            xc, yc = ct[0], ct[1]
            w1 = xc - x_min
            w2 = x_max - xc
            h1 = yc - y_min
            h2 = y_max - yc
            h = h1 + h2
            w = w1 + w2
            wh.append([w1,h1,w2,h2])
        else:
            h, w = y_max - y_min, x_max - x_min
            wh.append([w, h])
        radius = data_utils.gaussian_radius((math.ceil(h), math.ceil(w)))
        radius = max(0, int(radius))
        data_utils.draw_umich_gaussian(ct_hm, ct, radius)
        ct_ind.append(ct[1] * ct_hm.shape[1] + ct[0])
       

    def prepare_init(self, box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, h, w):
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        if cfg.ex_type == 'corner':
            img_init_poly = snake_text_utils.get_init(box, init_type='corner')
        else:
            img_init_poly = snake_text_utils.get_init(box)
        
        img_init_poly = snake_text_utils.uniformsample(img_init_poly, snake_config.init_poly_num)
        can_init_poly = snake_text_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = extreme_point
        can_gt_poly = snake_text_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        i_it_4pys.append(img_init_poly)
        c_it_4pys.append(can_init_poly)
        i_gt_4pys.append(img_gt_poly)
        c_gt_4pys.append(can_gt_poly)

    def prepare_evolution(self, poly, extreme_point, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys):
        if not cfg.use_otpg_flag:
            x_min, y_min = np.min(poly[:,0]), np.min(poly[:,1])
            x_max, y_max = np.max(poly[:,0]), np.max(poly[:,1])
            octagon = np.array([x_min,y_min,x_min,y_max,x_max,y_max,x_max,y_min])
            octagon = octagon.reshape(-1,2)
        else:
            x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
            x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])
            if cfg.ex_type == 'corner':
                octagon = extreme_point
            else:
                octagon = snake_text_utils.get_octagon(extreme_point)

        img_init_poly = snake_text_utils.uniformsample(octagon, snake_config.poly_num)
        can_init_poly = snake_text_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = snake_text_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)

        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_text_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)


        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)
    
    def prepare_shrunk_evolution(self, poly, extreme_point, img_init_shrunk_polys, can_init_shrunk_polys, img_gt_shrunk_polys, can_gt_shrunk_polys):
        def get_shrinked_poly(poly, shrink_ratio=0.4):
            d_i = cv2.contourArea(poly) * (1 - shrink_ratio) / cv2.arcLength(poly, True) + 0.5
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            shrinked_poly = np.array(pco.Execute(-d_i))
            if shrinked_poly != []:
                if len(shrinked_poly) == 1:
                    shrinked_poly = np.array(shrinked_poly[0]).reshape(-1,2)
                else:
                    poly_tmp = np.empty((0,2))
                    for spy in shrinked_poly:
                        spy = np.array(spy)
                        poly_tmp = np.vstack((poly_tmp, spy))
                    shrinked_poly = poly_tmp
            return shrinked_poly
        
        
        x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
        x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        octagon = snake_text_utils.get_octagon(extreme_point)

        img_init_poly = snake_text_utils.uniformsample(octagon, snake_config.poly_num)
        can_init_poly = snake_text_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)

        img_gt_poly = snake_text_utils.uniformsample(poly, len(poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_text_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)

    def prepare_neg_evolution(self, poly, instance_polys, img_init_polys, can_init_polys, img_gt_polys, can_gt_polys, inp_out_hw, img=None):
        from lib.sampling_neg_poly.gen_neg_samples import selecting_neg_poly, vis_bboxes, vis_polys
        out_h = inp_out_hw[2]
        out_w = inp_out_hw[3]
        sel_neg_poly = selecting_neg_poly(poly, instance_polys, out_h, out_w)
        
        if not cfg.use_otpg_flag:
            x_min, y_min = np.min(sel_neg_poly[:,0]), np.min(sel_neg_poly[:,1])
            x_max, y_max = np.max(sel_neg_poly[:,0]), np.max(sel_neg_poly[:,1])
            octagon = np.array([x_min,y_min,x_min,y_max,x_max,y_max,x_max,y_min])
            octagon = octagon.reshape(-1, 2)
        else:
            if cfg.ex_type == 'corner':
                corner_points = self.get_corner_points([[sel_neg_poly]])
                extreme_point = corner_points[0][0]
                octagon = extreme_point
            else:
                extreme_point = snake_text_utils.get_extreme_points(sel_neg_poly)
                octagon = snake_text_utils.get_octagon(extreme_point)
            x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
            x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])

        img_init_poly = snake_text_utils.uniformsample(octagon, snake_config.poly_num)
        can_init_poly = snake_text_utils.img_poly_to_can_poly(img_init_poly, x_min, y_min, x_max, y_max)
        img_gt_poly = snake_text_utils.uniformsample(sel_neg_poly, len(sel_neg_poly) * snake_config.gt_poly_num)
        tt_idx = np.argmin(np.power(img_gt_poly - img_init_poly[0], 2).sum(axis=1))
        img_gt_poly = np.roll(img_gt_poly, -tt_idx, axis=0)[::len(poly)]
        can_gt_poly = snake_text_utils.img_poly_to_can_poly(img_gt_poly, x_min, y_min, x_max, y_max)

        img_init_polys.append(img_init_poly)
        can_init_polys.append(can_init_poly)
        img_gt_polys.append(img_gt_poly)
        can_gt_polys.append(can_gt_poly)


    def prepare_merge(self, is_id, cls_id, cp_id, cp_cls):
        cp_id.append(is_id)
        cp_cls.append(cls_id)

    def __getitem__(self, index):
        #index = 10
        ann = self.anns[index]
        anno, path, img_id = self.process_info(ann)

       
        img, instance_polys, cls_ids = self.read_original_data(anno, path)
        raw_instance_polys = [poly for poly in instance_polys]

       

        height, width = img.shape[0], img.shape[1]
        orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw = \
            snake_text_utils.augment(
                img, self.split,
                snake_config.data_rng, snake_config.eig_val, snake_config.eig_vec,
                snake_config.mean, snake_config.std, instance_polys
            )
      
        instance_polys = self.transform_original_data(instance_polys, flipped, width, trans_output, inp_out_hw)
        instance_polys = self.get_valid_polys(instance_polys, inp_out_hw)
        
        if cfg.ex_type == 'corner':
            extreme_points = self.get_corner_points(instance_polys)
        else:
            extreme_points = self.get_extreme_points(instance_polys)

       
        # detection
        output_h, output_w = inp_out_hw[2:]
        ct_hm = np.zeros([cfg['heads']['ct_hm'], output_h, output_w], dtype=np.float32)
        if 'rs_hm' in cfg['heads']:
            rs_hm = np.zeros([cfg['heads']['rs_hm'], output_h, output_w], dtype=np.float32)
        else:
            rs_hm = None
        #box
        wh, ct_reg, ct_cls, ct_ind = [], [], [], []
        # init
        i_it_4pys,c_it_4pys,i_gt_4pys,c_gt_4pys = [], [], [], []
        # evolution
        i_it_pys,c_it_pys,i_gt_pys,c_gt_pys = [],[],[],[]

        if 1: #shrink
            i_it_s_pys,c_it_s_pys,i_gt_s_pys, c_gt_s_pys = [],[],[],[]

        if cfg.poly_cls_branch:
            neg_i_it_pys,neg_c_it_pys,neg_i_gt_pys,neg_c_gt_pys = [],[],[],[]

        for i in range(len(anno)):
            cls_id = cls_ids[i]
            instance_poly = instance_polys[i]
            instance_points = extreme_points[i]
            for j in range(len(instance_poly)):
                poly = instance_poly[j]
                extreme_point = instance_points[j]

                if cfg.hbb_enclose_type == 'obb':
                    x_min, y_min = np.min(extreme_point[:, 0]), np.min(extreme_point[:, 1])
                    x_max, y_max = np.max(extreme_point[:, 0]), np.max(extreme_point[:, 1])
                else:
                    x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
                    x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
                bbox = [x_min, y_min, x_max, y_max]
                h, w = y_max - y_min + 1, x_max - x_min + 1
                if h <= 1 or w <= 1:
                    continue
                if cfg.train.ct_reg:
                    decode_box = self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind, ct_reg, rs_hm=rs_hm)
                    self.prepare_init(decode_box, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
                else:
                    if cfg.train.mask_center:
                        self.prepare_detection_(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind)
                    else:
                        self.prepare_detection(bbox, poly, ct_hm, cls_id, wh, ct_cls, ct_ind, rs_hm=rs_hm)
                    self.prepare_init(bbox, extreme_point, i_it_4pys, c_it_4pys, i_gt_4pys, c_gt_4pys, output_h, output_w)
                self.prepare_evolution(poly, extreme_point, i_it_pys, c_it_pys, i_gt_pys, c_gt_pys)
                
                if 0:
                    self.prepare_shrunk_evolution(poly, extreme_point, i_it_s_pys, c_it_s_pys, i_gt_s_pys, c_gt_s_pys)
        
                if cfg.poly_cls_branch:
                    self.prepare_neg_evolution(poly, instance_polys,
                                               neg_i_it_pys, neg_c_it_pys, 
                                               neg_i_gt_pys, neg_c_gt_pys,
                                               inp_out_hw,
                                               img = orig_img)
                if i >= 40:                 
                    print('Anno-num:',len(anno))
        
        if 'rs_hm' in cfg['heads']:
            rs_hm = (rs_hm >= 1).astype(np.float32)

        ret = {'inp': inp}
        detection = {'ct_hm': ct_hm, 'wh': wh, 'reg': ct_reg, 'ct_cls': ct_cls, 'ct_ind': ct_ind, 'rs_hm': rs_hm}
        init = {'i_it_4py': i_it_4pys, 'c_it_4py': c_it_4pys, 'i_gt_4py': i_gt_4pys, 'c_gt_4py': c_gt_4pys}
        evolution = {'i_it_py': i_it_pys, 'c_it_py': c_it_pys, 'i_gt_py': i_gt_pys, 'c_gt_py': c_gt_pys}
        
        if cfg.poly_cls_branch:
            neg_evolution = {'neg_i_it_py': neg_i_it_pys, 'neg_c_it_py': neg_c_it_pys, 
                             'neg_i_gt_py': neg_i_gt_pys, 'neg_c_gt_py': neg_c_gt_pys }
            ret.update(neg_evolution) 
        
        ret.update(detection)
        ret.update(init)
        ret.update(evolution)
        #visualize_utils.visualize_snake_detection(orig_img, ret)
        #visualize_utils.visualize_snake_evolution(orig_img, ret)
        #exit()

        ct_num = len(ct_ind)
        meta = {'center': center, 'scale': scale, 'img_id': img_id, 'ann': ann, 'ct_num': ct_num}
        ret.update({'meta': meta})

        return ret

    def __len__(self):
        return len(self.anns)

