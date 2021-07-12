import numpy as np 
import cv2,json
from PIL import Image
import os,sys
import argparse
import matplotlib.pyplot as plt 

def parsing_det_file(_file, score_flag=False):
    with open(_file, 'r') as fid:
        det_polys_lst = []
        scores_lst    = []
        lines = fid.readlines()
        for line in lines:
            line = line.strip()
            vec = line.split(',')
            if score_flag:
                pts = list(map(int, vec[:-1]))
                score = int(vec[-1])
            else:
                pts = list(map(int, vec))
                score = 0.999
            pts = np.array(pts)
            pts_X = pts[0:len(pts):2]
            pts_Y = pts[1:len(pts):2]
            pts_X = pts_X[::-1]
            pts_Y = pts_Y[::-1]
            cat_pts_xy = np.hstack((pts_X.reshape(-1,1), pts_Y.reshape(-1,1)))
            if 0:
                I = np.zeros((250,250))
                plt.imshow(I)
                plt.plot(cat_pts_xy[:,0], cat_pts_xy[:,1])
                plt.plot(cat_pts_xy[0,0], cat_pts_xy[0,1], 'y+')
                plt.plot(cat_pts_xy[10,0], cat_pts_xy[10,1], 'yo')
                plt.show()
                exit()
            pts = cat_pts_xy.flatten()
            det_polys_lst.append(pts)
            scores_lst.append(score)
        return det_polys_lst, scores_lst
def vis_polys(img, polys_lst):
    plt.imshow(img)
    for k in range(len(polys_lst)):
        poly = polys_lst[k]
        poly_pts = poly.reshape(-1,2)
        plt.plot(poly_pts[:,0], poly_pts[:,1],'r', linewidth=2.5)
    return plt

def creating_submittion_format(dets_lst, scores_lst):
    dets_scores_lst = []
    for k in range(len(dets_lst)):
        poly_score_dict = dict()
        poly  = np.round(dets_lst[k])
        if len(poly) < 6:
            continue
        poly = poly.reshape(-1, 2).astype(np.int32).tolist()
        score = float(scores_lst[k])
        poly_score_dict['points'] = poly
        poly_score_dict['confidence'] = score
        dets_scores_lst.append(poly_score_dict)
    return dets_scores_lst


def parse_args():
    parser = argparse.ArgumentParser(description='Creating ArT submit format')
    parser.add_argument('--imgs_dir', help='imgs dir', default='data/art/test/image', type=str)
    parser.add_argument('--dets_dir', help='dets dir', default='cvpr21_results/PRB_v100/art/art/det_results', type=str)
    parser.add_argument('--submit_file', help='submit file', default='art_submit.json', type=str)
    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()
    imgs_dir = args.imgs_dir
    dets_dir = args.dets_dir
    submit_file = args.submit_file

    imgs_lst = sorted(os.listdir(imgs_dir))
    dets_lst = sorted(os.listdir(dets_dir))
    print("imgs_lst.len:", len(imgs_lst))
    print("dets_lst.len:", len(dets_lst))
    
    all_dets_dict = dict()
    for k in range(len(imgs_lst)):
        print("~~~~~~~~~~~~~~~~~~~~~Processing {}/{}:{}~~~~~~~~~~~~~~~~~~~~~~".format(k, len(imgs_lst), imgs_lst[k]))
        #if '3251' not in imgs_lst[k]:
        #    continue
        img_name = imgs_lst[k]
        img_id = img_name[:-4]
        img_fn = os.path.join(imgs_dir, img_name)
        #img = Image.open(img_fn)
        #img = np.array(img)
        img = cv2.imread(img_fn)
        print("img.shape:", img.shape)
        det_file = os.path.join(dets_dir,(img_name[:-4]+'.txt'))
        det_polys_lst, det_scores_lst = parsing_det_file(det_file)
        print("det_polys_lst.len", len(det_polys_lst))

        polys_scores_lst = creating_submittion_format(det_polys_lst, det_scores_lst)
        all_dets_dict['res_%s'%img_id[3:]] = polys_scores_lst
        if 0:
            plt = vis_polys(img, det_polys_lst)
            plt.show()
    with open(submit_file, 'w') as fid:
        fid.write(json.dumps(all_dets_dict))




