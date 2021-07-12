# --------------------------------------------------------
# Deformable Convolutional Networks
# Copyright (c) 2016 by Contributors
# Copyright (c) 2017 Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Sergey Karayev
# Modified by Yuwen Xiong, from from py-faster-rcnn (https://github.com/rbgirshick/py-faster-rcnn)
# --------------------------------------------------------

cimport cython
import cv2
import numpy as np
cimport numpy as np

DTYPE = np.float
ctypedef np.float_t DTYPE_t
DTYPE_INT = np.int
ctypedef np.int_t DTYPE_INT_t
DTYPE_UINT = np.uint8
ctypedef np.uint8_t DTYPE_UINT_t

def get_valid_full_mask_cython(
        np.ndarray[DTYPE_t, ndim=2] det_boxes,
        np.ndarray[DTYPE_t, ndim=3] det_masks,
        np.ndarray[DTYPE_INT_t, ndim=1] image_size,
        np.float mask_thresh = 0.5,
        np.float box_min_size = 3.0,
        np.float min_mask_area = 8.0):
    
    cdef unsigned int N = len(det_boxes)
    cdef np.ndarray[DTYPE_INT_t, ndim=1] bbox
    cdef np.ndarray[DTYPE_t, ndim=2]  mask
    cdef np.ndarray[DTYPE_t, ndim=2] full_mask
    
    new_dets = []
    new_masks = []
    for k in range(N):
        bbox = det_boxes[k,:4].astype(np.int)
        if (bbox[2]- bbox[0] <= box_min_size) or (bbox[3] - bbox[1] <= box_min_size):
            continue
        mask = det_masks[k, :, :]
        mask = cv2.resize(mask, (bbox[2] - bbox[0], (bbox[3] - bbox[1])), interpolation=cv2.INTER_LINEAR)
        mask[mask >  mask_thresh] = 1
        mask[mask <= mask_thresh] = 0
        if np.sum(mask) < min_mask_area:
            continue    
        full_mask = np.zeros((image_size[0], image_size[1]))
        full_mask[bbox[1]: bbox[3], bbox[0]: bbox[2]] = mask

        new_dets.append(det_boxes[k, :])
        new_masks.append(full_mask.astype(np.uint8))
    
    return new_dets, new_masks

def bbox_poly_overlaps_cython(
        np.ndarray[DTYPE_INT_t, ndim=2] boxes,
        np.ndarray[DTYPE_INT_t, ndim=2] polys,
        np.float width_thresh = 3.0, 
        np.float over_thresh  = 0.5):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    polys: (K, 500)  ndarray of float, filled with -1
    ----------
    Returns
    overlaps: (N, K) ndarray of overlap between boxes and polys
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = polys.shape[0]
    cdef np.ndarray[DTYPE_UINT_t, ndim=2] tmp_im
    #cdef np.ndarray[DTYPE_INT_t, ndim=2] valid_contour_pts
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef np.ndarray[DTYPE_INT_t, ndim=3] poly_crops = np.zeros((N, K, 4), dtype=DTYPE_INT)
    cdef DTYPE_INT_t iw, ih, crop_box_area, contour_w, b_xmin, b_ymin, b_xmax, b_ymax
    
    cdef DTYPE_t ua
    cdef DTYPE_t tmp_iou = 0.0
    cdef unsigned int k, n

    #poly_contour_lst = []
    for k in range(K):
        tmp_im = np.zeros((512, 512), dtype=DTYPE_UINT)
        cv2.fillPoly(tmp_im, [polys[k][3: 3 + polys[k][2]].reshape(-1, 2)], 1)  # polys[k][2] : the number of points in  k-th poly
        #_, contours, _ = cv2.findContours(tmp_im, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # contours:list of Nx1x2, N is the number of contour points
        contours, _ = cv2.findContours(tmp_im, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # contours:list of Nx1x2, N is the number of contour points
        if len(contours) == 0:
            continue
        contour_w = np.max(contours[0][:, 0, 0]) - np.min(contours[0][:, 0, 0]) 
        for n in range(N):
            inds = np.where((contours[0][:, 0, 0] >= boxes[n,0]) & 
                            (contours[0][:, 0, 0] <= boxes[n,2]))[0]

            valid_contour_pts = contours[0][inds,0,:]

            if valid_contour_pts.shape[0] >= 4:
                b_xmin = np.min(valid_contour_pts[:,0])
                b_ymin = np.min(valid_contour_pts[:,1])
                b_xmax = np.max(valid_contour_pts[:,0])
                b_ymax = np.max(valid_contour_pts[:,1])
                crop_box_area = (b_xmax - b_xmin + 1) * (b_ymax - b_ymin + 1)
                if (b_xmax - b_xmin) < contour_w/width_thresh:
                    continue 

                iw = min(boxes[n,2], b_xmax) - max(boxes[n,0], b_xmin) + 1
                if iw > 0:
                    ih = min(boxes[n,3], b_ymax) - max(boxes[n,1], b_ymin) + 1
                    if ih > 0:
                        ua = float((boxes[n,2]-boxes[n,0]+1)*(boxes[n,3]-boxes[n,1]+1) + crop_box_area - iw * ih)
                        overlaps[n,k] = iw * ih / ua
                        poly_crops[n, k, :] = np.array([b_xmin, b_ymin, b_xmax, b_ymax])

    return overlaps, poly_crops

def bbox_poly_overlaps_cython_vv(
        np.ndarray[DTYPE_INT_t, ndim=2] boxes,
        np.ndarray[DTYPE_INT_t, ndim=2] polys,
        np.float width_thresh = 3.0, 
        np.float over_thresh  = 0.5):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    polys: (K, 500)  ndarray of float, filled with -1
    ----------
    Returns
    overlaps: (N, K) ndarray of overlap between boxes and polys
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = polys.shape[0]
    cdef np.ndarray[DTYPE_UINT_t, ndim=2] tmp_im
    #cdef np.ndarray[DTYPE_INT_t, ndim=2] valid_contour_pts
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef np.ndarray[DTYPE_INT_t, ndim=3] poly_crops = np.zeros((N, K, 4), dtype=DTYPE_INT)
    cdef DTYPE_INT_t iw, ih, crop_box_area, contour_w, b_xmin, b_ymin, b_xmax, b_ymax
    
    cdef DTYPE_t ua
    cdef unsigned int k, n

    poly_contour_lst = []
    valid_anchors_lst = []
    valid_crop_lst = []
    for k in range(K):
        tmp_im = np.zeros((512, 512), dtype=DTYPE_UINT)
        cv2.fillPoly(tmp_im, [polys[k][3: 3 + polys[k][2]].reshape(-1,2)], 1)  # polys[k][2] : the number of points in  k-th poly
        #_, contours, _ = cv2.findContours(tmp_im, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # contours:list of Nx1x2, N is the number of contour points
        contours, _ = cv2.findContours(tmp_im, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE) # contours:list of Nx1x2, N is the number of contour points
        contour_w = np.max(contours[0][:, 0, 0]) - np.min(contours[0][:, 0, 0]) 
        for n in range(N):
            inds = np.where((contours[0][:, 0, 0] >= boxes[n,0]) & 
                            (contours[0][:, 0, 0] <= boxes[n,2]))[0]

            valid_contour_pts = contours[0][inds,0,:]

            if valid_contour_pts.shape[0] >= 4:
                b_xmin = np.min(valid_contour_pts[:,0])
                b_ymin = np.min(valid_contour_pts[:,1])
                b_xmax = np.max(valid_contour_pts[:,0])
                b_ymax = np.max(valid_contour_pts[:,1])
                crop_box_area = (b_xmax - b_xmin + 1) * (b_ymax - b_ymin + 1)
                if (b_xmax - b_xmin) < contour_w/width_thresh:
                    continue 

                iw = min(boxes[n,2], b_xmax) - max(boxes[n,0], b_xmin) + 1
                if iw > 0:
                    ih = min(boxes[n,3], b_ymax) - max(boxes[n,1], b_ymin) + 1
                    if ih > 0:
                        ua = float((boxes[n,2]-boxes[n,0]+1)*(boxes[n,3]-boxes[n,1]+1) + crop_box_area - iw * ih)
                        overlaps[n,k] = iw * ih / ua
                        if overlaps[n,k] > over_thresh:
                            poly_contour_lst.append(contours[0])
                            valid_anchors_lst.append(boxes[n])
                            valid_crop_lst.append([b_xmin, b_ymin, b_xmax, b_ymax])
    return poly_contour_lst, valid_anchors_lst, valid_crop_lst

def bbox_overlaps_cython(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(
                        (boxes[n, 2] - boxes[n, 0] + 1) *
                        (boxes[n, 3] - boxes[n, 1] + 1) +
                        box_area - iw * ih
                    )
                    overlaps[n, k] = iw * ih / ua
    return overlaps

def ignore_overlaps_cython(
        np.ndarray[DTYPE_t, ndim=2] boxes,
        np.ndarray[DTYPE_t, ndim=2] query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    cdef unsigned int N = boxes.shape[0]
    cdef unsigned int K = query_boxes.shape[0]
    cdef np.ndarray[DTYPE_t, ndim=2] overlaps = np.zeros((N, K), dtype=DTYPE)
    cdef DTYPE_t iw, ih, box_area
    cdef DTYPE_t ua
    cdef unsigned int k, n
    for k in range(K):
        box_area = (
            (query_boxes[k, 2] - query_boxes[k, 0] + 1) *
            (query_boxes[k, 3] - query_boxes[k, 1] + 1)
        )
        for n in range(N):
            iw = (
                min(boxes[n, 2], query_boxes[k, 2]) -
                max(boxes[n, 0], query_boxes[k, 0]) + 1
            )
            if iw > 0:
                ih = (
                    min(boxes[n, 3], query_boxes[k, 3]) -
                    max(boxes[n, 1], query_boxes[k, 1]) + 1
                )
                if ih > 0:
                    ua = float(box_area)
                    overlaps[n, k] = iw * ih / ua

    return overlaps
