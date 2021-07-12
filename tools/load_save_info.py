import numpy as np 
import scipy.io
import os, cv2
import matplotlib.pyplot as plt 
from shapely.geometry import *

def load_tot_gt_label(_file):
    data = scipy.io.loadmat(_file)
    gt_labels = data['polygt']

    polygon_gts = []
    for i in range(0,len(gt_labels)):
        X = gt_labels[i,1][0]
        Y = gt_labels[i,3][0]
        assert X.shape[0] == Y.shape[0],'Points_x should be equal Points_y'
        pts = np.vstack((X,Y)).transpose()
        pts_lst = pts.flatten()
        polygon_gts.append(pts_lst)
    return polygon_gts

def load_ctw_gt_label(_file):
    #reading bboxes_with_info from file
    fid = open(_file)
    contents = fid.read()
    lines =  contents.split('\n')
    bboxes_with_info = []
    for line in lines:
        vec = line.split(',')
        if len(vec) > 1: 
            vec = list(map(int,vec))
            bboxes_with_info.append(vec)
    bboxes_with_info = np.array(bboxes_with_info)

    #coverting bboxes_with_info to abs_polygons
    bboxes = bboxes_with_info[:,:4] #circum:(xmin,ymin,xmax,ymax)
    pts_offset_info = bboxes_with_info[:,4:]

    left_top_pts = bboxes[:,:2]
    repeat_lt_pts = np.tile(left_top_pts,14)
    polygons = repeat_lt_pts + pts_offset_info

    return polygons

def load_msra_gt_label(_file):
    with open(_file, 'r') as fid:
        polys_lst = []
        lines = fid.readlines()
        for line in lines:
            line = line.strip()
            vec = line.split(',')
            poly = list(map(int, vec[:-1]))
            polys_lst.append(np.array(poly))
        return polys_lst

def ctw_write_cords_into_file(dets,scores,_file):
    assert len(dets)==len(scores),'Checking dets_cords.size & dets_scores.size'
    img_id = os.path.basename(_file)[:-4]

    fid = open(_file,'w')
    for i in xrange(0,len(dets)):
        line = dets[i]
        if len(line) < 3:
            continue
        line = str(line)
        line = line.replace('[','').replace(']','')
        line = img_id + ',' + str(scores[i])+',' + line + '\n'

        fid.write(line)
    fid.close()

def tot_write_cords_into_file(det_contours, scores, _file):
    assert len(det_contours)==len(scores),'Checking dets_cords.size & dets_scores.size'
    
    polygon_dets = []
    for i in xrange(0,len(det_contours)):
        each_contour = np.array(det_contours[i])
        pt_pairs = each_contour.reshape(-1,2)
        pt_score = np.tile(scores[i],(len(pt_pairs),1))
        pt_pairs = np.hstack((pt_pairs, pt_score))
        polygon_dets.append(pt_pairs)
    scipy.io.savemat(_file, mdict={'accuInf':polygon_dets})
    return

def mlt_write_cords_into_file(img, bboxes,_file):
    assert bboxes.shape[1]==5 or bboxes.shape[1]==9
    
    fid = open(_file,'w')
    if bboxes.shape[1]==9:
        for box in bboxes:
            fid.write('%d,%d,%d,%d,%d,%d,%d,%d,%f\r\n'%
                (box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7],box[8]))
    else:
        h = img.shape[0]
        w = img.shape[1]
        for box in bboxes:
            xmin = max(1,box[0])
            ymin = max(1,box[1])
            xmax = min(w-1,box[2])
            ymax = min(h-1,box[3])
            score = box[4]
            fid.write('%d,%d,%d,%d,%d,%d,%d,%d,%f\r\n'%
                (xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax,score))
    fid.close()
    return

def icdar15_write_cords_into_file(img, bboxes,_file):
    assert bboxes.shape[1]==4 or bboxes.shape[1]==8
    
    h = img.shape[0]
    w = img.shape[1]
    fid = open(_file,'w')
    
    if bboxes.shape[1]==8:
        for box in bboxes:
            fid.write('%d,%d,%d,%d,%d,%d,%d,%d\r\n'%
                (box[0],box[1],box[2],box[3],box[4],box[5],box[6],box[7]))
    else:
        for box in bboxes:
            xmin = max(1,box[0])
            ymin = max(1,box[1])
            xmax = min(w-1,box[2])
            ymax = min(h-1,box[3])
            fid.write('%d,%d,%d,%d,%d,%d,%d,%d\r\n'%
                (xmin,ymin,xmax,ymin,xmax,ymax,xmin,ymax))
    
    fid.close()
    return


def icdar13_write_cords_into_file(img, bboxes, _file):
    assert bboxes.shape[1]==4 or bboxes.shape[1]==8

    if bboxes.shape[1]==8:
        bboxes_4d = np.zeros((bboxes.shape[0],4))
        X = bboxes[:,0:8:2]
        Y = bboxes[:,1:8:2]

        bboxes_4d[:,0::4] = X.min(axis=1)[:,np.newaxis]
        bboxes_4d[:,1::4] = Y.min(axis=1)[:,np.newaxis]
        bboxes_4d[:,2::4] = X.max(axis=1)[:,np.newaxis]
        bboxes_4d[:,3::4] = Y.max(axis=1)[:,np.newaxis]
        bboxes = bboxes_4d

    h = img.shape[0]
    w = img.shape[1]
    fid = open(_file,'w')
    for box in bboxes:
        xmin = max(1,box[0])
        ymin = max(1,box[1])
        xmax = min(w-1,box[2])
        ymax = min(h-1,box[3])
        fid.write('%d,%d,%d,%d\r\n'%(xmin,ymin,xmax,ymax))

    fid.close()
    return





def coco_write_boxes_into_file(img,bboxes,_file):
    assert bboxes.shape[1]==5 or bboxes.shape[1]==9

    if bboxes.shape[1]==9:
        bboxes_8d = bboxes[:,:8]
        scores = bboxes[:,-1].reshape(-1,1)

        bboxes_4d = np.zeros((bboxes.shape[0],4))
        X = bboxes_8d[:,0:8:2]
        Y = bboxes_8d[:,1:8:2]

        bboxes_4d[:,0::4] = X.min(axis=1)[:,np.newaxis]
        bboxes_4d[:,1::4] = Y.min(axis=1)[:,np.newaxis]
        bboxes_4d[:,2::4] = X.max(axis=1)[:,np.newaxis]
        bboxes_4d[:,3::4] = Y.max(axis=1)[:,np.newaxis]
        bboxes = np.hstack((bboxes_4d,scores))


    h = img.shape[0]
    w = img.shape[1]
    fid = open(_file,'w')
    for box in bboxes:
        xmin = max(1,box[0])
        ymin = max(1,box[1])
        xmax = min(w-1,box[2])
        ymax = min(h-1,box[3])
        score = box[4]
        fid.write('%d,%d,%d,%d,%f\r\n'%(xmin,ymin,xmax,ymax,score))
    
    fid.close()
    return

#----------------------------------Used in PCR------------------------------------#
def resorting_corner_pts(points, order='anticlockwise'):
    '''
    params: points 4x2
    '''
    center_x = np.mean(points[:,0])
    center_y = np.mean(points[:,1])
    left_ids = np.where(points[:,0]<= center_x)[0]
    left_points = points[left_ids]
    right_ids = np.where(points[:,0] > center_x)[0]
    right_points = points[right_ids]

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

    if order == 'clockwise':
        pp = np.hstack((p1,p2,p3,p4)).reshape(-1,2) 
    else:
        pp = np.hstack((p1,p4,p3,p2)).reshape(-1,2) #anticlockwise
    return pp

def saving_mot_det_results(_file, polygons, testing_set='ic15', img=None):
    with open(_file, 'w') as fid:
        for k in range(len(polygons)):
            poly = polygons[k]
            poly = np.round(poly).astype(np.int32)
            rect = cv2.minAreaRect(poly) 
            new_poly = cv2.boxPoints(rect)
            new_poly = resorting_corner_pts(new_poly, order='clockwise')
            new_poly = np.round(new_poly).astype(np.int)
            new_poly = list(new_poly.flatten())
            str_poly = list(map(str, new_poly))
            if testing_set == 'mlt':
                str_poly = ','.join(str_poly)+","+str(0.9999)+'\r\n'  #x1,y1,x2,y2,x3,y3,x4,y4,score
            elif testing_set == 'ic15':
                str_poly = ','.join(str_poly)+'\r\n'
            elif testing_set == 'msra':
                str_poly = ','.join(str_poly)+'\r\n'
            else:
                raise ValueError('Not supported testing_set:{}'.format(testing_set))
            fid.write(str_poly)

def saving_det_results(_file, polygons, img=None):
    with open(_file, 'w') as fid:
        for k in range(len(polygons)):
            poly = polygons[k]
            poly = np.round(poly).astype(np.int32)
            if 1: # Correct the invalid detected polygons
                #tuple_poly = tuple([tuple(e) for e in poly])
                polygon_obj = Polygon(poly)
                if not polygon_obj.is_valid:
                    print("the det is invalid in {}".format(os.path.basename(_file)))
                    rect = cv2.minAreaRect(poly) 
                    new_poly = cv2.boxPoints(rect)
                    new_poly = resorting_corner_pts(new_poly)
                    new_poly = np.round(new_poly).astype(np.int)
                    #new_poly = remove_close_points(poly)
                    h, w, _ = img.shape
                    #new_poly[0,:] = np.clip(new_poly[0,:], 0, w-1)
                    #new_poly[1,:] = np.clip(new_poly[1,:], 0, h-1)
                    poly = new_poly

            poly = list(poly.flatten())
            str_poly = list(map(str, poly))
            str_poly = ','.join(str_poly)+'\n'
            fid.write(str_poly)

