from shapely.geometry import Polygon, MultiPolygon
from shapely.ops import cascaded_union, polygonize
import numpy as np
from lib.utils.snake import snake_config

DEBUG = False
def get_shape_poly(poly):
    poly = poly.astype(np.int32)
    shape_poly = Polygon(poly)
    if shape_poly.is_valid:
        return shape_poly

    # self-intersected situation
    linering = shape_poly.exterior

    # disassemble polygons from multiple line strings
    mls = linering.intersection(linering)
    # assemble polygons from multiple line strings
    polygons = polygonize(mls)
    multi_shape_poly = MultiPolygon(polygons)
    return multi_shape_poly

def poly_iou(poly1, poly2):
    #poly = cascaded_union([poly1, poly2])
    #union = poly.area
    #intersection = poly1.area + poly2.area - union
    #iou = intersection / union
    try:
        union = poly1.intersection(poly2)
        iou = max(union.area/poly1.area, union.area/poly2.area)
    except Exception as e:  
        iou = 0
    return iou

def get_poly_iou_matrix(poly1, poly2):
    poly1 = [get_shape_poly(poly) for poly in poly1]
    poly2 = [get_shape_poly(poly) for poly in poly2]
    iou_matrix = np.zeros([len(poly1), len(poly2)])
    for i in range(len(poly1)):
        for j in range(len(poly2)):
            iou_matrix[i, j] = poly_iou(poly1[i], poly2[j])
    return iou_matrix

def get_poly_match_ind(poly1, poly2):
    iou_matrix = get_poly_iou_matrix(poly1, poly2)
    iou = iou_matrix.max(axis=1)
    gt_ind = iou_matrix.argmax(axis=1)
    poly_ind = np.argwhere(iou > snake_config.poly_iou).ravel()
    gt_ind = gt_ind[poly_ind]
    return poly_ind, gt_ind

def poly_nms(poly):
    if DEBUG:
        print('-----------------Poly-nms--------------------------')
        print('polygons.num:', len(poly))
        print('poly_iou:', snake_config.poly_iou)
    iou_matrix = get_poly_iou_matrix(poly, poly)
    iou_matrix[np.arange(len(poly)), np.arange(len(poly))] = 0
    overlapped = np.zeros([len(poly)])
    poly_ = []
    ind = []
    for i in range(len(poly)):
        if overlapped[i]:
            continue
        poly_.append(poly[i])
        ind.append(i)
        overlapped[iou_matrix[i] > snake_config.poly_iou] = 1
    return poly_, ind
