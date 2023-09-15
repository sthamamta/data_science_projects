## TSNE PLot 

import time 
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D 
import seaborn as sns 
import matplotlib.patheffects as PathEffects 
import numpy as np 
from sklearn import preprocessing 

def draw_plot(emb_df, y, plot_file): 

    time_start = time.time() 
    num_classes=7 
    tsne_df = emb_df 
    tsne_labels = y 
    tsne = TSNE() 
    tsne_results = tsne.fit_transform(tsne_df.values) 
    #print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    le = preprocessing.LabelEncoder() 
    labels = le.fit_transform(tsne_labels) 
    tsne_df['label'] = labels 
    name_classes = list(le.classes_)
    
    plot_scatter(tsne_results, tsne_df['label'], name_classes, plot_file)


def plot_scatter(x, colors, name_classes, plot_file): 
    # choose a color palette with seaborn. 
    num_classes = len(np.unique(colors)) 
    palette = np.array(sns.color_palette("hls", num_classes)) 
    # print(palette)

    # create a scatter plot. 
    f = plt.figure(figsize=(8, 8)) 
    ax = plt.subplot(aspect='equal') 
    sc = ax.scatter(x[:,0], x[:,1],  c=palette[colors.astype(np.int)], cmap=plt.cm.get_cmap('Paired')) 
    plt.xlim(-25, 25) 
    plt.ylim(-25, 25) 
    ax.axis('off') 
    ax.axis('tight') 
    plt.savefig(plot_file) 

    # add the labels for each group 
    txts = [] 
    for i in range(num_classes): 
    # Position of each label at median of data points. 
        xtext, ytext = np.median(x[colors == i, :], axis=0) 
        txt = ax.text(xtext, ytext, str(name_classes[i]), fontsize=10) 
        txt.set_path_effects([ 
                PathEffects.Stroke(linewidth=5, foreground="w"), 
                PathEffects.Normal()]) 
        txts.append(txt) 

    return f, ax, sc, txts 

 
