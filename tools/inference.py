import os,time,cv2,glob
import json,tqdm
import pyclipper
import numpy as np
import torch
import torch.utils.data as data

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

import matplotlib.pyplot as  plt
import pycocotools.mask as maskUtils

DEBUG = True
class Dataset(data.Dataset):
    def __init__(self):
        super(Dataset, self).__init__()
        if DEBUG:
            print('demo_path:', cfg.demo_path)
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

def poly_rle_nms(img, polys, scores, nms_thresh=0.3):
    assert len(scores) ==  len(polys), 'poly.num != scores.num'
    org_h, org_w, _ = img.shape
    rles = snake_eval_utils.coco_poly_to_rle(polys, org_h, org_w)
    print('encoding rles done.')
    iscrowd = [int(0) for k in range(len(all_boundaries))]
    ious = maskUtils.iou(rles, rles, iscrowd)
    print('ious:', ious)

    order = scores.argsort()[::-1]
    keep = []
    while order.size>0:
        i = order[0]
        keep.append(i)
        ovr = ious[i][order[1:]].astype(np.float32)
        inds = np.where(ovr <= nms_thresh)[0]
        order = order[inds+1]
    return keep

def remove_close_points(points):
    next_points = np.concatenate((points[1:,:], points[0][None]))
    diff = np.min(np.abs(points - next_points), axis=-1)
    idx = diff > 1e-5
    new_points = points[idx]
    return new_points

def visualizing_det_results(img, polygons, _file, scores=None, poly_scores=None):
    plt.imshow(img[:,:,(2,1,0)])
    cnt = 0
    for k in range(len(polygons)):
        instance_pts = polygons[k]
        X = instance_pts[:, 0]
        Y = instance_pts[:, 1]
        
        pdet = Polygon(instance_pts)
        if not pdet.is_valid:
            print('saving intersecting polygon...')
            cnt = cnt + 1
            fn = _file[:-4]+'_'+str(cnt)+'.png'
            plt.imshow(img[:,:,(2,1,0)])
            plt.plot(X,Y, 'r', linewidth=1)
            plt.savefig(fn)
            plt.close() 
        '''
        plt.plot(X,Y, 'r', linewidth=2)
        if scores is not None:
            #plt.text(X[0], Y[0],'s:{:.2f}'.format(scores[k]), bbox=dict(fc='g',alpha=0.5))
            xc = np.mean(X)
            yc = np.mean(Y)
            if poly_scores is not None:
                plt.text(xc, yc,'{:.2f}/{:.2f}'.format(scores[k], poly_scores[k]), fontdict={'color':'yellow','size':8})
            else:
                plt.text(xc, yc,'{:.2f}'.format(scores[k]), fontdict={'color':'red','size':8})
        '''
    #plt.savefig(_file)
    #plt.close()

def vis_tmp_results(org_img, detections, ex_points, all_boundaries, contour_feat, poly_scores, output, indx=None):
    tmp_dir = 'ffff_tmp_vis_dir'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)

    raw_ct_hm = output['ct_hm'].detach().cpu().numpy()
    nsm_ct_hm  = output['nms_ct_hm'].detach().cpu().numpy()
    pred_wh = output['wh'].detach().cpu().numpy()
    dets = output['detection'].detach().cpu().numpy()
    K = len(all_boundaries)
    plt.subplot(2,2,1)
    plt.imshow(raw_ct_hm[0,0,...])
    plt.subplot(2,2,2)
    plt.imshow(nsm_ct_hm[0,0,...])
    for i in range(len(dets)):
        xmin,ymin,xmax,ymax = dets[i,:4]
        poly = np.array([xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax,xmin,ymin]).reshape(-1,2)
        plt.plot(poly[:,0], poly[:,1], 'y')
        
    plt.subplot(2,2,3)
    plt.imshow(org_img)
    for i in range(K):
        box = detections[i,:4]
        box_score = detections[i,4]
        ex = ex_points[i]
        ex = np.round(ex).astype(np.int32)

        ex = np.concatenate((ex, ex[0].reshape(-1,2)), axis=0)
        box_8d = np.array([box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]])
        box_8d = box_8d.reshape(-1, 2)
        box_8d = np.concatenate((box_8d, box_8d[0].reshape(-1,2)), axis=0)
        poly = all_boundaries[i]
        poly = np.round(poly).astype(np.int32)

        poly_score = poly_scores[i]
        plt.plot(poly[:,0], poly[:,1],'r')
        #plt.plot(poly[0,0], poly[0,1], 'yo')
        #plt.plot(poly[10,0], poly[10,1], 'y*')
        plt.plot(ex[:,0], ex[:,1], 'b-')
        #plt.text(ex[0,0], ex[0,1],'1')
        #plt.text(ex[1,0], ex[1,1],'2')
        #plt.text(ex[2,0], ex[2,1],'3')
        #plt.text(ex[3,0], ex[3,1],'4')
        plt.plot(box_8d[:,0], box_8d[:,1],'g--')
        plt.text(poly[0,0], poly[0,1], '{:.2f}/{:.2f}'.format(box_score, poly_score), color='r')

    #Feat normalizing
    sum_feat = np.sum(contour_feat, axis=1)
    min_sfeat = np.min(sum_feat, axis=-1, keepdims=True)
    max_sfeat = np.max(sum_feat, axis=-1, keepdims=True)
    norm_sfeat = (sum_feat - min_sfeat)/(max_sfeat-min_sfeat)
    plt.subplot(2,2,4)
    I = np.zeros((org_img.shape[0], org_img.shape[1]))
    h, w = I.shape
    for i in range(K):
        poly = np.round(all_boundaries[i]).astype(np.int32)
        feat_act = norm_sfeat[i]
        for j in range(len(poly)):
            I[np.clip(poly[j,1],0,h-1), np.clip(poly[j,0],0,w-1)]=feat_act[j]
    plt.imshow(I)
    vis_file = os.path.join(tmp_dir, (indx+'.png'))
    plt.show()
    #plt.savefig(vis_file)
    #plt.close()

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

def rescoring_polygons(polygons, score_map):
    def get_shrinked_poly(poly, shrink_ratio=0.4):
        d_i = cv2.contourArea(poly) * (1 - shrink_ratio) / cv2.arcLength(poly, True) + 0.5
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = np.array(pco.Execute(-d_i))

        if 0:
            import matplotlib.pyplot as plt
            plt.imshow(score_map)
            plt.plot(poly[:,0], poly[:,1])
            if len(shrinked_poly) > 1:
                for py in shrinked_poly:
                    py = np.array(py)
                    plt.plot(py[:,0],py[:,1],'r+')
                    plt.plot(py[:,0],py[:,1],'r')
            plt.savefig('ff.png')
            plt.close()
            if len(shrinked_poly) > 1:
                exit()

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
    def get_shrinked_poly_v1(poly, shrink_ratio=0.4):
        from shapely.geometry import Polygon
        polygon_shape = Polygon(poly)
        distance = polygon_shape.area * (1 - np.power(shrink_ratio, 2)) / polygon_shape.length
        subject = [tuple(l) for l in poly]
        padding = pyclipper.PyclipperOffset()
        padding.AddPath(subject, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        shrinked_poly = padding.Execute(-distance)

        if 0:
            import matplotlib.pyplot as plt
            plt.imshow(score_map)
            plt.plot(poly[:,0], poly[:,1])
            if len(shrinked_poly) > 1:
                for py in shrinked_poly:
                    py = np.array(py)
                    plt.plot(py[:,0],py[:,1],'r+')
                    plt.plot(py[:,0],py[:,1],'r')
            plt.savefig('ffc.png')
            plt.close()
            if len(shrinked_poly) > 1:
                print('shrinked_poly:',shrinked_poly)
                exit()

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
    #-------------------------------------------------------------------#
    score_map = score_map[0,0,...]
    rescores_lst = []
    for k in range(len(polygons)):
        poly = polygons[k]
        shrinked_poly = get_shrinked_poly_v1(poly)
        if len(shrinked_poly)==0:
            s = 0.0
        else:
            shrinked_poly = shrinked_poly.astype(np.int32)
            vals = score_map[(shrinked_poly[:,1],shrinked_poly[:,0])]
            s = np.mean(vals)
        rescores_lst.append(s)
    return np.array(rescores_lst)

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

        if DEBUG:
            print('------------------img_name={}-------------------------'.format(img_name))
            print('org_img.shape:', org_img.shape)
            print('rz_img.shape:',  rz_img.shape)
            print('input-size:({}, {})'.format(h,w))
        
        if cfg.rescore_map_flag:
            rs_thresh = 0.6
            detections = output['detection'].detach().cpu().numpy()
            polys = output['py'][-1].detach().cpu().numpy()
            rs_hm = torch.sigmoid(output['rs_hm']).detach().cpu().numpy()
            if 0:
                print('output.keys:', output.keys())

            rescores = rescoring_polygons(polys, rs_hm)
            conf_keep = np.where(rescores > rs_thresh)[0]
            
            detections = detections[conf_keep]
            pys = [polys[k]* snake_config.down_ratio for k in conf_keep]
            rescores = rescores[conf_keep]        
            
            rs_hm_path = os.path.join(cfg.vis_dir,(img_name[:-4]+'_rs.png'))
            import matplotlib.pyplot as plt
            plt.imshow(rs_hm[0,0,...])
            plt.savefig(rs_hm_path)
            if 0:
                print('detections.shape:', detections.shape)
                print('pys.num:', len(pys))
                print('rs_hm.shape:', rs_hm.shape)
                x = rs_hm[0,0,...]
                import matplotlib.pyplot as plt 
                plt.imshow(x)
                for k in range(len(pys)):
                    plt.plot(pys[k][:,0], pys[k][:, 1])
                plt.savefig('{}.png'.format(img_name[:-4]))
                plt.close()
                np.save('rs_hm.npy', x)
                np.save('pys.npy', np.array(pys))
                exit()
        else:
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

            if cfg.vis_intermediate_output != 'none':
                if cfg.vis_intermediate_output == 'htp':
                    xmin,ymin,xmax,ymax = bboxes[:,0::4], bboxes[:,1::4], bboxes[:, 2::4], bboxes[:,3::4]
                    pys = np.hstack((xmin,ymin, xmin,ymax,xmax,ymax,xmax,ymin))
                    pys = pys.reshape(pys.shape[0],4,2)
                elif cfg.vis_intermediate_output == 'otp':
                    pys = ex_pts
                elif cfg.vis_intermediate_output == 'clm_1':
                    pys = iter_ply_output_lst[0]
                elif cfg.vis_intermediate_output == 'clm_2':
                    pys = iter_ply_output_lst[1]
                else:
                    raise ValueError('Not supported type:', cfg.vis_intermediate_output)
                cfg.poly_cls_branch = False


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
                if DEBUG:
                    print('py_cls_scores:', text_poly_scores)

            if DEBUG:
                print('dets_num:', len(pys))

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
            
            if len(all_boundaries) != 0:
                detections[:,:4] /= rz_ratio
                ex_points /= rz_ratio
                all_boundaries = [poly/rz_ratio for poly in all_boundaries]
                
            if 0:
                import matplotlib.pyplot as plt
                nms_polygons,rem_inds = snake_poly_utils.poly_nms(all_boundaries)
                print('nms_polygons.num:', len(nms_polygons))
                plt.subplot(1,2,1)
                plt = plot_poly(org_img, all_boundaries,scores=scores)
                plt.subplot(1,2,2)
                plt = plot_poly(org_img, nms_polygons)
                plt.savefig('a.png')
                exit()
            
            #nms
            all_boundaries, rem_inds = snake_poly_utils.poly_nms(all_boundaries)
            detections = detections[rem_inds]
            ex_points = ex_points[rem_inds]
            final_contour_feat = final_contour_feat[rem_inds]
            if cfg.poly_cls_branch:
                poly_scores = poly_scores[rem_inds]
            pp_used_time = time.time() - pp_time_s
            infer_time_lst.append([net_used_time, pp_used_time])
            if DEBUG:
                print('infer_time:',[net_used_time, pp_used_time])

            if 0:
                vis_tmp_results(org_img, detections, ex_points, all_boundaries, final_contour_feat, poly_scores, output, indx=img_name[:-4])

        #--------------------------------saving results-------------------------------#
        if cfg.testing_set == 'mlt':
            det_file = os.path.join(cfg.det_dir, ('res_'+img_name[3:-4]+'.txt'))
            saving_mot_det_results(det_file, all_boundaries, testing_set=cfg.testing_set, img=org_img)
        elif cfg.testing_set == 'ic15':
            det_file = os.path.join(cfg.det_dir, ('res_'+img_name[:-4]+'.txt'))
            saving_mot_det_results(det_file, all_boundaries, testing_set=cfg.testing_set, img=org_img)
        elif cfg.testing_set == 'msra':
            det_file = os.path.join(cfg.det_dir, ('res_'+img_name[:-4]+'.txt'))
            saving_mot_det_results(det_file, all_boundaries, testing_set=cfg.testing_set, img=org_img)
        else: #for arbitrary-shape datasets, e.g., CTW,TOT,ART
            det_file = os.path.join(cfg.det_dir, (img_name[:-4]+'.txt'))
            saving_det_results(det_file, all_boundaries, img=org_img)
        
        continue        
        #------------------------visualizing results---------------------------------#
        ## ~~~~~~ vis-v0 ~~~~~~~ ##
        vis_file = os.path.join(cfg.vis_dir,(img_name[:-4]+'.png'))
        if cfg.testing_set == 'ctw':
            gt_file = os.path.join(cfg.gts_dir, (img_name[:-4]+'.txt'))
            gt_polys = load_ctw_gt_label(gt_file)
        elif cfg.testing_set == 'tot':
            gt_file = os.path.join(cfg.gts_dir, ('poly_gt_'+img_name[:-4]+'.mat'))
            gt_polys = load_tot_gt_label(gt_file)
        elif cfg.testing_set == 'art':
            gt_polys = None
        elif cfg.testing_set == 'msra':
            gt_file = os.path.join(cfg.gts_dir, ('gt_'+img_name[:-4]+'.txt'))
            gt_polys = load_msra_gt_label(gt_file)
        else:
            raise ValueError('Not supported dataset ({}) for visualizing'.format(cfg.testing_set))
        plt = vis_dets_gts(org_img, all_boundaries, gt_polys)
        plt.savefig(vis_file,dpi=600,format='png')
        plt.close()
        ### ~~~~~~~~~ vis-v1 ~~~~~~~~~~~ ###
        # if cfg.poly_cls_branch:
        #     visualizing_det_results(org_img,all_boundaries,vis_file, scores=detections[:,4],poly_scores=poly_scores)
        # else:
        #     visualizing_det_results(org_img,all_boundaries,vis_file, scores=detections[:,4])
        ## vis-v2
        #hm_vis_dir = os.path.join(cfg.vis_dir, ('../vis_hm_on_img_dir'))
        #if not os.path.exists(hm_vis_dir):
        #    os.makedirs(hm_vis_dir)
        #visualizer.visualize(output, batch, os.path.join(hm_vis_dir,(img_name[:-4]+'.png')))

    np.save('infer_time.npy', np.array(infer_time_lst))
