import os, sys
from os import listdir
from scipy import io
import numpy as np
from polygon_fast import iou
from polygon_fast import iod

DEBUG = False
"""
Input format: y0,x0, ..... yn,xn. Each detection is separated by the end of line token ('\n')'
"""

input_dir =  sys.argv[1] #detection directory goes here
gt_dir = sys.argv[2]   #gt directory goes here
eval_cache = sys.argv[3]
fid_path = sys.argv[4] #eval result file fid
conf_thresh = float(sys.argv[5])
eval_iou_thresh = float(sys.argv[6])


allInputs = sorted(listdir(input_dir))

def input_reading_mod(input_dir, input):
    """This helper convert input"""
    with open('%s/%s' %(input_dir, input), 'r') as input_fid:
        pred = input_fid.readlines()
    det = [x.strip('\n') for x in pred]
    return det


def gt_reading_mod(gt_dir, gt_id):
    gt_id = gt_id.split('.')[0]
    gt = io.loadmat('%s/poly_gt_%s.mat' %(gt_dir, gt_id))
    gt = gt['polygt']
    return gt

def detection_filtering(detections, groundtruths, threshold=0.5):
    for gt_id, gt in enumerate(groundtruths):
        if (gt[5] == '#') and (gt[1].shape[1] > 1):
            gt_x = list(map(int, np.squeeze(gt[1])))
            gt_y = list(map(int, np.squeeze(gt[3])))
            for det_id, detection in enumerate(detections):
                detection = detection.split(',')
                detection = list(map(int, detection[2:]))
                #det_y = detection[0::2]
                #det_x = detection[1::2]
                det_x = detection[0::2]
                det_y = detection[1::2]
                det_gt_iou = iod(det_x, det_y, gt_x, gt_y)
                if det_gt_iou > threshold:
                    detections[det_id] = []

            detections[:] = [item for item in detections if item != []]
    return detections

global_tp = 0
global_fp = 0
global_fn = 0
global_num_of_gt = 0
global_num_of_det = 0

for k in range(len(allInputs)):
    input_id = allInputs[k]
    if (input_id != '.DS_Store'):
        if DEBUG:
            print("~~~~~~~~~~~~~~~{}/{}:{}~~~~~~~~~~~~~~~~~~".format(k, len(allInputs), input_id))
        #print(input_id)
        detections = input_reading_mod(input_dir, input_id)
        groundtruths = gt_reading_mod(gt_dir, input_id)
        detections = detection_filtering(detections, groundtruths) #filtering detections overlaps with DC area
        dc_id = np.where(groundtruths[:, 5] == '#')
        groundtruths = np.delete(groundtruths, (dc_id), (0))
    
        global_num_of_gt = global_num_of_gt + groundtruths.shape[0]
        global_num_of_det = global_num_of_det + len(detections)
        iou_table = np.zeros((len(detections), groundtruths.shape[0]))
        det_flag = np.zeros((len(detections), 1))
        gt_flag = np.zeros((groundtruths.shape[0], 1))
        tp = 0
        fp = 0
        fn = 0
        if DEBUG:
            print("dets.num:{}, gts_num:{}".format(len(detections), len(groundtruths)))

        if len(detections) > 0:
            for det_id, detection in enumerate(detections):
                detection = detection.split(',')
                detection = list(map(int, detection[2:]))
                #det_y = detection[0::2]
                #det_x = detection[1::2]
                det_x = detection[0::2]
                det_y = detection[1::2]
                if len(groundtruths) > 0:
                    for gt_id, gt in enumerate(groundtruths):
                        gt_x = list(map(int, np.squeeze(gt[1])))
                        gt_y = list(map(int, np.squeeze(gt[3])))
                        iou_table[det_id, gt_id] = iou(det_x, det_y, gt_x, gt_y)
                        
                        if 0:
                            import matplotlib
                            matplotlib.use('Agg')
                            import matplotlib.pyplot as plt
                            xx = np.zeros((1500, 1000))
                            plt.imshow(xx)
                            plt.plot(det_x, det_y,'r')
                            plt.plot(gt_x, gt_y, 'g')
                            plt.savefig('t.png')
                            plt.close()

                    best_matched_gt_id = np.argmax(iou_table[det_id, :]) #identified the best matched detection candidates with current groundtruth
                    if (iou_table[det_id, best_matched_gt_id] >= eval_iou_thresh):
                        if gt_flag[best_matched_gt_id] == 0: ### if the gt is already matched previously, it should be a false positive
                            tp = tp + 1.0
                            global_tp = global_tp + 1.0
                            gt_flag[best_matched_gt_id] = 1
                        else:
                            fp = fp + 1.0
                            global_fp = global_fp + 1.0
                    else:
                        fp = fp + 1.0
                        global_fp = global_fp + 1.0

        try:
            local_precision = tp / (tp + fp)
        except ZeroDivisionError:
            local_precision = 0

        try:
            local_recall = tp / groundtruths.shape[0]
        except ZeroDivisionError:
            local_recall = 0
        
        fid = open(os.path.join(eval_cache, 'per_img_iou_eval.txt'), 'a')
        temp = ('%s | Precision: %s, Recall: %s\n' %(input_id, str(local_precision), str(local_recall)))
        if DEBUG:
            print(temp)
            print("eval_iou_thresh:", eval_iou_thresh)
        fid.write(temp)
        fid.close()
            
global_precision = global_tp / global_num_of_det
global_recall = global_tp / global_num_of_gt
f_score = 2*global_precision*global_recall/(global_precision+global_recall)

fid = open(fid_path, 'a')
temp = ('IOUeval(conf=%s) | Precision: %s, Recall: %s, F-score: %s\n' %(str(conf_thresh), str(global_precision), str(global_recall), str(f_score)))
fid.write(temp)
fid.close()
