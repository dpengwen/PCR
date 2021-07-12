from lib.utils import img_utils, data_utils
from lib.utils.snake import snake_config
import matplotlib.pyplot as plt
import numpy as np
import torch
from itertools import cycle
import os
from PIL import Image

mean = snake_config.mean
std = snake_config.std

DEBUG = True
class Visualizer:
    def visualize_ex(self, output, batch):
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio

        fig, ax = plt.subplots(1, figsize=(20, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color)

        plt.show()

    def visualize_training_box(self, output, batch):
        import cv2 
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        h, w, _ = inp.shape
        box = output['detection'][:, :4].detach().cpu().numpy() * snake_config.down_ratio
        #box = output['detection'][:, :4].detach().cpu().numpy()
        nms_ct_hm = output['nms_ct_hm'].detach().cpu().numpy()
        nms_ct_hm = nms_ct_hm[0,0,...]
        nms_ct_hm = Image.fromarray(nms_ct_hm)
        nms_ct_hm = nms_ct_hm.resize((w, h))


        ex = output['py']
        ex = ex[-1] if isinstance(ex, list) else ex
        ex = ex.detach().cpu().numpy() * snake_config.down_ratio
        #ex = ex.detach().cpu().numpy()

        fig, ax = plt.subplots(1, figsize=(10, 10))
        fig.tight_layout()
        ax.axis('off')
        ax.imshow(inp)
        ax.imshow(np.array(nms_ct_hm))

        colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
        np.random.shuffle(colors)
        colors = cycle(colors)
        for i in range(len(ex)):
            color = next(colors).tolist()
            poly = ex[i]
            poly = np.append(poly, [poly[0]], axis=0)
            ax.plot(poly[:, 0], poly[:, 1], color=color, linewidth=1)

            x_min, y_min, x_max, y_max = box[i]
            ax.plot([x_min, x_min, x_max, x_max, x_min], [y_min, y_max, y_max, y_min, y_min], color='w', linewidth=0.5)

            print("box:", box[i])

        plt.show()

    def visualize_heatmap_on_img(self, output, batch, vis_file=None):
        import cv2 
        inp = img_utils.bgr_to_rgb(img_utils.unnormalize_img(batch['inp'][0], mean, std).permute(1, 2, 0))
        h, w, _ = inp.shape
        raw_ct_hm = output['ct_hm'].detach().cpu().numpy()
        raw_ct_hm = raw_ct_hm[0,0,...]
        raw_ct_hm = Image.fromarray(raw_ct_hm)
        raw_ct_hm = raw_ct_hm.resize((w, h))
        nms_ct_hm = output['nms_ct_hm'].detach().cpu().numpy()
        nms_ct_hm = nms_ct_hm[0,0,...]
        nms_ct_hm = Image.fromarray(nms_ct_hm)
        nms_ct_hm = nms_ct_hm.resize((w, h))

        raw_ct_hm = np.array(raw_ct_hm)
        plt.imshow(inp)
        plt.imshow(raw_ct_hm, alpha=0.8, cmap='rainbow')
        plt.axis('off')
        if vis_file is not None:
            plt.savefig(vis_file,format='png',dpi=600)
            plt.close()
        else:
            plt.show()
            

    def visualize(self, output, batch, vis_file=None):
        # self.visualize_ex(output, batch)
        #self.visualize_training_box(output, batch)
        self.visualize_heatmap_on_img(output, batch, vis_file=vis_file)

