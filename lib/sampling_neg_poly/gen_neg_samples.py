import numpy as np 
import random
import pickle
import matplotlib.pyplot as plt 
import matplotlib.patches as patches
from .bbox import bbox_overlaps_cython

def selecting_anchors(anchors, max_overlaps):
    #specific ranges
    sel_anchors_lst = []
    threshs_lst = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    
    #inds = np.where(max_overlaps <= threshs_lst[0])[0]
    #sel_anchors_lst.append(anchors[inds])

    inds = np.where((threshs_lst[0] < max_overlaps) &
                                    (max_overlaps <= threshs_lst[1]))
    sel_anchors_lst.append(anchors[inds])

    inds = np.where((threshs_lst[1] < max_overlaps) &
                                    (max_overlaps <= threshs_lst[2]))
    sel_anchors_lst.append(anchors[inds])

    inds = np.where((threshs_lst[2] < max_overlaps) &
                                    (max_overlaps <= threshs_lst[3]))
    sel_anchors_lst.append(anchors[inds])

    inds = np.where((threshs_lst[3] < max_overlaps) &
                                    (max_overlaps <= threshs_lst[4]))
    sel_anchors_lst.append(anchors[inds])

    inds = np.where((threshs_lst[4] < max_overlaps) &
                                    (max_overlaps <= threshs_lst[5]))
    sel_anchors_lst.append(anchors[inds])

    inds = np.where((threshs_lst[6] < max_overlaps) &
                                    (max_overlaps <= threshs_lst[6]))
    sel_anchors_lst.append(anchors[inds])

    #randomly choose one anchor
    sel_anchor = np.empty((0,4))
    for k in range(len(sel_anchors_lst)):
        t = sel_anchors_lst[k]
        if len(t) == 0:
            continue
        ind = random.randrange(len(t))
        sel_anchor = t[ind,:].reshape(1, -1)
        break

    if len(sel_anchor) == 0:
        ind = np.argmin(max_overlaps)
        sel_anchor = anchors[ind].reshape(1,-1)

    return sel_anchor

def vis_bboxes(bboxes, img, color='g'):
    plt.imshow(img)
    for k in range(len(bboxes)):
        box = bboxes[k]
        xmin,ymin,xmax,ymax = box
        points = [xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax,xmin,ymin]
        plt.plot(points[0:len(points):2], points[1:len(points):2], color)
    return plt

def vis_polys(polys_lst, img, color='g'):
    plt.imshow(img)
    for poly in polys_lst:
            poly = np.append(poly, [poly[0]], axis=0)
            plt.plot(poly[:, 0], poly[:, 1], color)

    return plt
def polys_to_bboxes(polys):
    bboxes = []
    for i in range(len(polys)):
            poly = polys[i]
            for j in range(len(poly)):
                    ply = poly[j]
                    x_min, y_min = np.min(ply[:, 0]), np.min(ply[:, 1])
                    x_max, y_max = np.max(ply[:, 0]), np.max(ply[:, 1])
                    bboxes.append([x_min, y_min, x_max, y_max])
    return np.array(bboxes)

def get_anchors(base_anchor, out_h, out_w, window_stride = 8):
    shift_x = np.arange(0, out_w/window_stride) * window_stride
    shift_y = np.arange(0, out_h/window_stride) * window_stride
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
    K = shifts.shape[0]

    base_anchor_n = 1
    anchors = base_anchor.reshape(1, base_anchor_n, 4) + shifts.reshape(1, K, 4).transpose((1,0,2))
    anchors = anchors.reshape(K * base_anchor_n, 4)

    inside_inds = np.where((anchors[:, 0] >= 0) &
                                               (anchors[:, 1] >= 0) &
                                               (anchors[:, 2] <= out_w + 32) &
                                               (anchors[:, 3] <= out_h + 32))[0] 

    anchors = anchors[inside_inds, :]

    return anchors

def selecting_neg_poly(poly, instance_polys, out_h=128, out_w=128):
    gt_bboxes = polys_to_bboxes(instance_polys)

    x_min, y_min = np.min(poly[:, 0]), np.min(poly[:, 1])
    x_max, y_max = np.max(poly[:, 0]), np.max(poly[:, 1])
    base_box = [x_min - x_min, y_min - y_min, x_max - x_min, y_max - y_min]
    base_box = np.array(base_box).reshape(1, -1)
    anchors = get_anchors(base_box, out_h, out_w)
    overlaps = bbox_overlaps_cython(np.ascontiguousarray(gt_bboxes).astype(float),
                                    np.ascontiguousarray(anchors).astype(float))
    max_overlaps = np.max(overlaps, axis=0)
    sel_anchor = selecting_anchors(anchors, max_overlaps)
    sel_poly = np.hstack((sel_anchor.flatten()[0] + poly[:, 0::2] - x_min, 
                          sel_anchor.flatten()[1] + poly[:, 1::2] - y_min))
    sel_poly[:, 0] = np.clip(sel_poly[:, 0], 0, out_w - 1)
    sel_poly[:, 1] = np.clip(sel_poly[:, 1], 0, out_h - 1)
    
    return sel_poly




