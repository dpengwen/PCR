import numpy as np 
import matplotlib.pyplot as plt 

def plot_poly(img,polygons,color='r',scores=None):
    plt.imshow(img[:,:,(2,1,0)])
    for k in range(len(polygons)):
        instance_pts = polygons[k]
        X = instance_pts[:, 0]
        Y = instance_pts[:, 1]
        plt.plot(X,Y, color, linewidth=2)
        if scores is not None:
            #plt.text(X[0], Y[0],'s:{:.2f}'.format(scores[k]), bbox=dict(fc='g',alpha=0.5))
            plt.text(X[0], Y[0],'s:{:.2f}'.format(scores[k]))
    return plt

def vis_dets_gts(img,dets,gts=None,scores=None):
    plt.imshow(img[:,:,(2,1,0)])
    if gts is not None:
        for k in range(len(gts)):
            pts = gts[k].reshape(-1,2)
            pts = np.concatenate((pts,pts[0].reshape(-1,2)),axis=0)
            plt.plot(pts[:,0], pts[:,1], 'g', linewidth=2.5)
    for k in range(len(dets)):
        pts = dets[k]
        pts = np.concatenate((pts,pts[0].reshape(-1,2)),axis=0)
        plt.plot(pts[:,0], pts[:,1], 'r', linewidth=2.5)
        if scores is not None:
            #plt.text(X[0], Y[0],'s:{:.2f}'.format(scores[k]), bbox=dict(fc='g',alpha=0.5))
            plt.text(pts[0,0], pts[0,1],'s:{:.2f}'.format(scores[k]))
    plt.axis('off')
    return plt
