# -*- coding: utf-8 -*-
"""
Created on Tue May 17 08:42:00 2016

@author: atreo
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from PIL import Image


def pascal_classes():
  classes = {'aeroplane' : 1,  'bicycle'   : 2,  'bird'        : 3,  'boat'         : 4, 
             'bottle'    : 5,  'bus'       : 6,  'car'         : 7,  'cat'          : 8, 
             'chair'     : 9,  'cow'       : 10, 'diningtable' : 11, 'dog'          : 12, 
             'horse'     : 13, 'motorbike' : 14, 'person'      : 15, 'potted-plant' : 16, 
             'sheep'     : 17, 'sofa'      : 18, 'train'       : 19, 'tv-monitor'   : 20}

  return classes

def pascal_palette():
  palette = {(  0,   0,   0) : 0 ,
             (128,   0,   0) : 1 ,
             (  0, 128,   0) : 2 ,
             (128, 128,   0) : 3 ,
             (  0,   0, 128) : 4 ,
             (128,   0, 128) : 5 ,
             (  0, 128, 128) : 6 ,
             (128, 128, 128) : 7 ,
             ( 64,   0,   0) : 8 ,
             (192,   0,   0) : 9 ,
             ( 64, 128,   0) : 10,
             (192, 128,   0) : 11,
             ( 64,   0, 128) : 12,
             (192,   0, 128) : 13,
             ( 64, 128, 128) : 14,
             (192, 128, 128) : 15,
             (  0,  64,   0) : 16,
             (128,  64,   0) : 17,
             (  0, 192,   0) : 18,
             (128, 192,   0) : 19,
             (  0,  64, 128) : 20 }

  return palette



def plot_inference_results(image, prediction):    
    
    # prepare prediction for overlay
    pred = Image.fromarray(cm.gist_ncar(prediction, bytes=True)).convert('RGBA')
    pred_copy = pred.copy()
    mask = pred_copy.convert("L").point(lambda x: min(x, 150))
    pred_copy.putalpha(mask)
    
    # overlay image with prediction
    merged = image.copy()
    merged.paste(pred_copy, (0, 0), pred_copy)
        
    plt.figure(1)
    
    plt.subplot(2,2,3)
    plt.imshow(image)
    plt.title('Image')
    
    plt.subplot(2,2,4)
    plt.imshow(pred, cmap=cm.gist_ncar)
    plt.title('Prediction')   
    
    plt.subplot(2,1,1)
    plt.imshow(merged)
    plt.title('Image + Prediction')
    
    plt.show()
