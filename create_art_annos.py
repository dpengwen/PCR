import numpy as np 
import os,sys, json
import cv2,shutil
import time
from shapely.geometry import Polygon
import matplotlib.pyplot as plt 
from PIL import Image

DEBUG = True

def get_instance_polygons(img_annos, remove_unreadble_flag=True):
    polygons = []
    for per_anno in img_annos:
        poly = per_anno['points']
        if remove_unreadble_flag:
        	illegibility = per_anno['illegibility']
        	if illegibility:
        		continue
        poly = np.array(poly).flatten()
        polygons.append(poly)
    print("remained_boxes:{}/{}".format(len(polygons),len(img_annos)))
    return polygons

def create_image_info_dict(img, ind, flag='2020'):
    height, width, _ = img.shape
    pid = '%04d'%(ind+1)
    img_id = int(flag+pid)
    file_name = flag+'_'+pid+'.jpg'
    img_info_dict = {"id":img_id,
                     "file_name": file_name,
                     "height": height,
                     "width": width
                     }
    return img_id, file_name, img_info_dict

def create_anno_info_dict(img_id, gt_box, idx):
    iscrowd = 0
    cid = 1
    ply_obj = Polygon(gt_box.reshape(-1,2))
    area = ply_obj.area
    xmin = np.min(gt_box[0:len(gt_box):2])
    ymin = np.min(gt_box[1:len(gt_box):2])
    xmax = np.max(gt_box[0:len(gt_box):2])
    ymax = np.max(gt_box[1:len(gt_box):2])
    h = ymax - ymin + 1
    w = xmax - xmin + 1
    hbox = [xmin, ymin, w, h]
    segmentation =[list(gt_box)]
    anno_dict = {"id": idx,
                 "image_id": img_id,
                 "category_id": cid,
                 "iscrowd": iscrowd,
                 "bbox": hbox,
                 "area": area,
                 "segmentation": segmentation}
    return anno_dict
def vis_annos(img, bboxes):
    plt.imshow(img)
    for k in range(len(bboxes)):
        box = bboxes[k].reshape(-1,2)
        pts = np.concatenate((box, box[0].reshape(-1,2)), axis=0)
        plt.plot(pts[:,0], pts[:,1],'r')
    return plt
if __name__ == '__main__':
    dataset_path = '.'
    train_imgs_dir = os.path.join(dataset_path,'ART/Detection/train_images')
    train_gts_cache  = os.path.join(dataset_path,'ART/Detection/train_labels.json')

    remove_unreadble_flag = True
    
    dst_img_dir = "train/images"
    dst_anno_file = "train/art_train_instance.json"
    if not os.path.exists(dst_img_dir):
        os.makedirs(dst_img_dir)
    
    with open(train_gts_cache, 'r') as fid:
        all_annos_dict = json.load(fid)

    instance_cnt = 0
    train_imgs_lst = sorted(os.listdir(train_imgs_dir))
    all_images_lst,all_annos_lst = [],[]
    for k in range(len(train_imgs_lst)):
        print('~~~~~~~~~~~~~~~~~~~~~ Processing {}/{}:{} ~~~~~~~~~~~~~~~~~~~~~~~'.format(k, len(train_imgs_lst), train_imgs_lst[k]))
        img_name = train_imgs_lst[k]
        img_full_name = os.path.join(train_imgs_dir, img_name)
        #img = cv2.imread(img_full_name)
        img = Image.open(img_full_name)
        img = np.array(img)
        print("img.shape:", img.shape)
        if len(img.shape) != 3:
        	img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        per_img_annos = all_annos_dict[img_name[:-4]]
        gt_bboxes = get_instance_polygons(per_img_annos, remove_unreadble_flag)

        if 0:
            plt = vis_annos(img, gt_bboxes)
            plt.show()
        img_id, img_new_name, img_info_dict = create_image_info_dict(img, k)
        all_images_lst.append(img_info_dict)
        dst_fn = os.path.join(dst_img_dir, img_new_name)
        shutil.copyfile(img_full_name, dst_fn)

        for j in range(len(gt_bboxes)):
            #print("-------------{}/{}------------".format(j, len(gt_bboxes)))
            instance_cnt += 1
            ann_info_dict = create_anno_info_dict(img_id, gt_bboxes[j], instance_cnt)
            all_annos_lst.append(ann_info_dict)
  
    print("all_images_lst.len:", len(all_images_lst))
    print("all_annos_lst.len:",  len(all_annos_lst))

    categories=[{"supercategory":"none", "id":1, "name": "text"}]
    final_annotations = dict()
    final_annotations[u"images"] = all_images_lst
    final_annotations[u"annotations"] = all_annos_lst
    final_annotations[u"categories"] = categories
    with open(dst_anno_file, 'w') as fid:
        json.dump(final_annotations, fid)


