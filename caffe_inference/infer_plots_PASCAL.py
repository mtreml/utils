import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
import matplotlib.cm as cm



"""
Maps classes to IDs
"""
def pascal_class2id():
    class2id = {'background': 0,
             'aeroplane' : 1,  'bicycle'   : 2,  'bird'        : 3,  'boat'         : 4, 
             'bottle'    : 5,  'bus'       : 6,  'car'         : 7,  'cat'          : 8, 
             'chair'     : 9,  'cow'       : 10, 'diningtable' : 11, 'dog'          : 12, 
             'horse'     : 13, 'motorbike' : 14, 'person'      : 15, 'potted-plant' : 16, 
             'sheep'     : 17, 'sofa'      : 18, 'train'       : 19, 'tv-monitor'   : 20}

    return class2id


"""
Maps IDs to classes
"""
def pascal_id2class():
    cl = pascal_class2id()
    id2class = {v: k for k, v in cl.items()}
    
    return id2class


"""
Maps rgba-colors to IDs
"""
def pascal_color2id():
    color2id = {(0,   0,   0, 255) : 0 ,
             (128,   0,   0, 255) : 1 ,
             (  0, 128,   0, 255) : 2 ,
             (128, 128,   0, 255) : 3 ,
             (  0,   0, 128, 255) : 4 ,
             (128,   0, 128, 255) : 5 ,
             (  0, 128, 128, 255) : 6 ,
             (128, 128, 128, 255) : 7 ,
             ( 64,   0,   0, 255) : 8 ,
             (192,   0,   0, 255) : 9 ,
             ( 64, 128,   0, 255) : 10,
             (192, 128,   0, 255) : 11,
             ( 64,   0, 128, 255) : 12,
             (192,   0, 128, 255) : 13,
             ( 64, 128, 128, 255) : 14,
             (192, 128, 128, 255) : 15,
             (  0,  64,   0, 255) : 16,
             (128,  64,   0, 255) : 17,
             (  0, 192,   0, 255) : 18,
             (128, 192,   0, 255) : 19,
             (  0,  64, 128, 255) : 20 }

    return color2id


"""
Maps IDs to rgba-colors
"""
def pascal_id2color():
    pal = pascal_color2id()
    id2color = {v: k for k, v in pal.items()}
    
    return id2color


"""
Normalizes RGBA colors to range [0, 1]
"""
def rgba(color):
    r,g,b,a = color
    return (r/255, g/255, b/255, a/255)


"""
Create colormap for PASCAL classes
"""
def create_cmap():

    # extract all colors from the PASCAL palette
    colordict = pascal_id2color()
    id2class = pascal_id2class()
    colorlist = np.array([list(rgba(colordict[i])) for i in range(len(colordict))], dtype=np.float32)
    print('COLORMAP HAS', len(colorlist), 'COLORS')
    # create the new map
    my_cmap = ListedColormap(colorlist)

    # define colorbar properties
    bounds = list(np.arange(len(colordict)+1)-0.5)
    norm = BoundaryNorm(bounds, my_cmap.N)
    class_id_labels = list(np.arange(len(colordict)))
    class_name_labels = list(id2class[i] for i in class_id_labels)
    class_id_name_labels = list(str(class_id_labels[i]) + ' (' + class_name_labels[i] + ')' for i in class_id_labels)
    
    return my_cmap, bounds, norm, class_id_labels, class_id_name_labels


"""
Creates a plot with 2 Subplots:

1) Groundtruth image VS overlay of groundtruth image and predicted segmentation

2) Groundtruth segmentation VS predicted segmentation
"""
def plot_inference_results(image, gt, out):    
    pred = Image.fromarray(np.uint8(out))
    unique_ids = list(np.unique(out))
    
    # create PASCAL colormap
    my_cmap, bounds, norm, class_id_labels, class_id_name_labels = create_cmap()    
    
    # image & image with prediction overlay
    plt.figure('Inference result')    
    plt.subplot(1,2,1)
    plt.imshow(image)
    plt.title('Image')    
    plt.subplot(1,2,2)
    plt.hold(True)
    plt.imshow(image)
    plt.imshow(pred, cmap=my_cmap, vmin=0, vmax=20, alpha=0.7)
    cbar = plt.colorbar(ticks=class_id_labels,
                 spacing='uniform',
                 boundaries=bounds,
                norm=norm)
    cbar.ax.set_yticklabels(class_id_name_labels)
    plt.title('Prediction') 
    plt.show()
    
    plt.figure('Gt vs prediction')    
    plt.subplot(1,2,2)
    plt.imshow(pred, cmap=my_cmap, vmin=0, vmax=20)
    cbar = plt.colorbar(ticks=class_id_labels,
                 spacing='uniform',
                 boundaries=bounds,
                norm=norm)
    cbar.ax.set_yticklabels(class_id_name_labels)
    plt.title( str(len(unique_ids)) + ' classes: ' + str(unique_ids) )    
    plt.subplot(1,2,1)
    plt.imshow(gt, cmap=my_cmap, vmin=0, vmax=20)  
    plt.title('Ground truth')
    plt.show()


"""
Print which classes were predicted
"""
def printClassPredInfo(out):
    id2class = pascal_id2class()
    unique_ids = list(np.unique(out))
    unique_classes = [id2class[id] for id in unique_ids]
    print(str(len(unique_ids)) + ' classes: ' + str(unique_ids) + ' or ' + str(unique_classes))
    
