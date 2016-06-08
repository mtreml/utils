import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colorbar import ColorbarBase
import matplotlib.cm as cm
import labels as CITYPalette







"""
Normalizes RGBA colors to range [0, 1]
"""
def rgba_CITY(color):
    r,g,b = color
    return (r/255, g/255, b/255, 1)



"""
Create colormap for CITYSCAPE classes
"""
def create_cmap():

    # extract all colors from the PASCAL palette
    # filter labels included in eval
    eval_labels = [ label for label in CITYPalette.labels if label.ignoreInEval==False ]
    colordict = [ label.color for label in eval_labels]
    colorlist = np.array([list(rgba_CITY(colordict[i])) for i in range(len(colordict))], dtype=np.float32)
    print('COLORMAP HAS', len(colorlist), 'COLORS')
    # create the new map
    my_cmap = ListedColormap(colorlist)

    # define colorbar properties
    bounds = list(np.arange(len(colordict)+1)-0.5)
    norm = BoundaryNorm(bounds, my_cmap.N)
    class_id_labels = list( label.trainId for label in eval_labels )
    class_name_labels = list( label.name for label in eval_labels )
    class_id_name_labels = list( str(class_id_labels[i]) + ' (' + class_name_labels[i] + ')' for i in range(len(class_id_labels)))

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
    plt.imshow(pred, cmap=my_cmap, vmin=0, vmax=18, alpha=0.7)
    cbar = plt.colorbar(ticks=class_id_labels,
                 spacing='uniform',
                 boundaries=bounds,
                norm=norm)
    cbar.ax.set_yticklabels(class_id_name_labels)
    plt.title('Prediction') 
    plt.show()
    
    plt.figure('Gt vs prediction')    
    plt.subplot(1,2,2)
    plt.imshow(pred, cmap=my_cmap, vmin=0, vmax=18)
    cbar = plt.colorbar(ticks=class_id_labels,
                 spacing='uniform',
                 boundaries=bounds,
                norm=norm)
    cbar.ax.set_yticklabels(class_id_name_labels)
    plt.title( str(len(unique_ids)) + ' classes: ' + str(unique_ids) )    
    plt.subplot(1,2,1)
    plt.imshow(gt, cmap=my_cmap, vmin=0, vmax=18)
#    cbar = plt.colorbar(ticks=class_id_labels,
#                 spacing='uniform',
#                 boundaries=bounds,
#                norm=norm)
#    cbar.ax.set_yticklabels(class_id_name_labels)  
    plt.title('Ground truth')
    plt.show()


"""
Print which classes were predicted
"""
def printClassPredInfo(out):
    eval_labels = [ label for label in CITYPalette.labels if label.ignoreInEval==False ]
    unique_ids = list(np.unique(out))
    unique_classes = [ label.name for label in eval_labels if label.trainId in unique_ids ]
    print(str(len(unique_ids)) + ' classes: ' + str(unique_ids) + ' or ' + str(unique_classes))
    
