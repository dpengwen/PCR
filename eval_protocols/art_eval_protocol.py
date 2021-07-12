import sys,os
import pickle as cPickle
import json
import cv2
import argparse
import numpy as np 
import matplotlib.pyplot as plt 
from shapely.geometry import *

DEBUG = True

def vis_polygon_on_img(im, polys, color='r', linewidth=3):
    plt.imshow(im)
    for k in range(len(polys)):
        poly = polys[k]
        poly_X = poly[0:len(poly):2]
        poly_Y = poly[1:len(poly):2]
        X = np.hstack((np.array(poly_X), poly_X[0]))
        Y = np.hstack((np.array(poly_Y), poly_Y[0]))
        plt.plot(X, Y, color, linewidth=linewidth)
    return plt

def read_file(_file):
  with open(_file,'r') as fid:
    lines = fid.readlines()
    objs = []
    for line in lines:
        vec = line.strip().split(',')
        vec = map(int,vec)
        objs.append(vec)
    objs = np.array(objs)
    return objs

def curve_parse_gt_txt(filename):
    """ Parse a ctw1500 txt file """
    with open(filename.strip(),'r') as f:
        gts = f.readlines()
        objects = []
        for obj in gts:
            cors = obj.strip().split(',')
            cors_num = len(cors)
            obj_struct = {}
            # class name
            obj_struct['name'] = 'text'
            obj_struct['difficult'] = 0
            obj_struct['bbox'] = [int(cors[i]) for i in range(cors_num)]
            objects.append(obj_struct)
            # assert(0), obj_struct['bbox']
    return objects
                  
def load_gt_set(gt_dir, gt_lst, cache_dir=None):
    gt_cache_file = os.path.join(cache_dir,'gt_cache.pkl')
    if not os.path.exists(gt_cache_file):
        gt_set = {}
        for i in range(0,len(gt_lst)):
            gt_name = gt_lst[i]
            index = gt_name[:-4]

            full_gt_file = os.path.join(gt_dir,gt_name)
            objs = curve_parse_gt_txt(full_gt_file)

            gt_set[index] = objs

        with open(gt_cache_file, 'wb') as fid:
            cPickle.dump(gt_set, fid, cPickle.HIGHEST_PROTOCOL)

    else:
        with open(gt_cache_file,'rb') as fid:
            gt_set = cPickle.load(fid) 

    npos = 0
    cls_gt_set = {}
    for (index, gt) in gt_set.items():
        R = [obj for obj in gt if obj['name'] == 'text'] # text 
    
        if not R: continue
        bbox = [x['bbox'] for x in R]
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        cls_gt_set[index] = {'bbox': bbox,'det': det}

    return npos, cls_gt_set

def load_det_set(det_dir, det_lst, cache_dir=None, conf_thresh=0.001):
    det_cache_file = os.path.join(cache_dir,'dets_cache.txt')
    print("det_lst.size:", len(det_lst))
    
    if not os.path.exists(det_cache_file):
        fid = open(det_cache_file,'w')
        for i in range(0, len(det_lst)):
            det_file = det_lst[i]
            full_det_file = os.path.join(det_dir, det_file)
            with open(full_det_file,'r') as f:
                lines = f.readlines()
                for line in lines:
                    flag_str = det_file[:-4]+','+str(0.999999)
                    new_line = flag_str+','+line
                    fid.write(new_line)
        fid.close()
    print("combine all dets-file done.")
    with open(det_cache_file, 'r') as f:
        lines = f.readlines()
   
    splitlines = [x.strip().split(',') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])
    
    #BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    BB = [[float(z) for z in x[2:]] for x in splitlines]
    
    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    sorted_scores = np.sort(-confidence)
    
    image_ids = [image_ids[x] for x in sorted_ind]
    #det_set = BB[sorted_ind, :]
    det_set = [BB[x] for x in sorted_ind]

    #----dpwen:remove some dets-bboxes via score-----------#
    keep_inds = np.where(sorted_scores <= -conf_thresh)[0]
    det_set = det_set[:len(keep_inds)]
    image_ids = image_ids[:len(keep_inds)]
    #det_set = det_set[keep_inds,:]
    #img_ids_tmp = image_ids
    #image_ids = [img_ids_tmp[x] for x in keep_inds]
    print("removed bboxes_num:",(len(sorted_ind) - len(keep_inds)))
   
    return image_ids, det_set
            
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_art(img_dir, gt_dir, det_dir, eval_res_dir, ovthresh=0.5, conf_thresh=0.001):
    gt_lst  = sorted(os.listdir(gt_dir))
    det_lst = sorted(os.listdir(det_dir))
    print("gt_lst_len:",  len(gt_lst))
    print("det_lst_len:", len(det_lst))

    assert len(gt_lst)==len(det_lst), 'gt_file number ({})  v.s. det_file number ({})'.format(len(gt_lst), len(det_lst))
    npos, gt_set  = load_gt_set(gt_dir, gt_lst, cache_dir=eval_res_dir)
    print("loading gt-set done.")
    img_inds, det_set = load_det_set(det_dir, det_lst, cache_dir=eval_res_dir, conf_thresh=conf_thresh)  
    print("loading det-set done.")

    if DEBUG:
        print("img_inds.len:", len(img_inds))
    print('Calculating...')
    nd = len(img_inds)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
        #print "@----------{}/{}:{}------------@".format(d+1, nd, img_inds[d])
        R = gt_set[img_inds[d]]
        
        #det_bbox = det_set[d, :].astype(float)
        det_bbox = det_set[d]

        pts_num = len(det_bbox)
        det_pts = tuple([(det_bbox[i], det_bbox[i+1]) for i in range(0, pts_num, 2)])
        pdet = Polygon(det_pts)
        
        

        if not pdet.is_valid: 
            if 0:
                print("det_pts", det_pts)
                img_full_name = os.path.join(img_dir, (img_inds[d]+'.jpg'))
                img = cv2.imread(img_full_name)
                det_pts_lst = list(np.array(det_pts).flatten())
                all_dets = [det_pts_lst]
                plt = vis_polygon_on_img(img, all_dets)
                plt.show()
            print('the det region (id={}) has intersecting sides.'.format(d))
            continue
            
        ovmax = -np.inf
        GTS = R['bbox']
        
        ls_pgt = [] 
        overlaps = np.zeros(len(GTS))
        for iix in range(len(GTS)):
            gt_x = GTS[iix]
            top_left_p = gt_x[:2]
            offsets = gt_x[4:]
            assert len(offsets)%2==0, 'offsets must be divided by 2!'
            pts_num = int(len(offsets)/2)
            if pts_num < 3:
                continue   
            abs_pl = np.array(top_left_p*pts_num)+np.array(offsets)
            pts = abs_pl.reshape(-1, 2).astype(np.float32)

            pgt = Polygon(pts)
            if not pgt.is_valid: 
                print('GT polygon has intersecting sides (image_id:{}).'.format(img_inds[d]))
                continue
            try:
                sec = pdet.intersection(pgt)
            except Exception as e:
                print(e)
                continue
            #assert(sec.is_valid), 'polygon has intersection sides.'
            if not sec.is_valid:
                print("the intersection is not a valid polygon (area:{})!".format(sec.area))


            if 0:
                if (not pdet.is_valid) or (not sec.is_valid):
                    print("img_id:", img_inds[d])
                    valid_pdet = pdet.is_valid
                    valid_sec = sec.is_valid
                    print("valid_pdet:{}, valide_sec:{}".format(valid_pdet, valid_sec))
                    #imgs_lst = os.listdir(img_dir)
                    img_full_name = os.path.join(img_dir, (img_inds[d]+'.jpg'))
                    img = cv2.imread(img_full_name)
                    det_pts_lst = list(np.array(det_pts).flatten())
                    gt_pts_lst  = list(np.array(pts).flatten())
                    all_dets = [det_pts_lst]
                    all_gts  = [gt_pts_lst]
                    plt = vis_polygon_on_img(img, all_dets)
                    plt = vis_polygon_on_img(img, all_gts, 'g')
                    plt.show()
                    #dst_file = os.path.join(eval_res_dir, ('{}.png'.format(d)))
                    #plt.savefig(dst_file)
                    #plt.close()
                


            inters = sec.area
            uni = pgt.area + pdet.area - inters
            overlaps[iix] = inters*1.0 / uni
            # ls_overlaps.append(inters*1.0 / uni)

        ovmax = np.max(overlaps)
        jmax  = np.argmax(overlaps)

        if ovmax > ovthresh:
            # if not R['difficult'][jmax]:
            if not R['det'][jmax]:
                tp[d] = 1.
                R['det'][jmax] = 1
            else:
                fp[d] = 1.
        else:
            fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric=False)
    
    recall = rec[-1]
    precision = prec[-1]

    f_score = 2.0/(1/rec[-1]+1/prec[-1])

    return ap, recall, precision, f_score


def parse_args():
    parser = argparse.ArgumentParser(description='eval using iou-protocol')
    parser.add_argument('--imgs_dir', help='imgs dir', default=None, type=str)
    parser.add_argument('--gts_dir', help='gts dir', default=None, type=str)
    parser.add_argument('--dets_dir', help='dets .txt', default=None, type=str)
    parser.add_argument('--eval_dir', help='tmp eval dir', default=None, type=str)
    parser.add_argument('--eval_res_file', help='eval_res_file', default=None, type=str)
    parser.add_argument('--eval_iou_thresh', help='evaluation iou threshold', default=0.5, type=float)
    parser.add_argument('--conf_thresh', help='confidence threshold', default=0.5, type=float)

    args = parser.parse_args()
    return args

if __name__== '__main__':
    args = parse_args()
    if not os.path.exists(args.eval_dir):
        os.makedirs(args.eval_dir)

    print("----------------------------------- Evaluating (iou_thresh={}) ----------------------------------".format(args.eval_iou_thresh))
    AP, Recall, Precision, F_score = eval_art(args.imgs_dir, args.gts_dir, args.dets_dir, args.eval_dir, args.eval_iou_thresh, conf_thresh=args.conf_thresh)
    
    print("Conf:{}, R:{}, P:{}, F:{}".format(args.conf_thresh, Recall, Precision, F_score))

    fid = open(args.eval_res_file, 'w')
    fid.write('IOU(conf={:.2f})　| Precision:{}, Recall:{}, F_score:{}\n'.format(args.conf_thresh,  Precision, Recall, F_score))
    fid.close()

    
