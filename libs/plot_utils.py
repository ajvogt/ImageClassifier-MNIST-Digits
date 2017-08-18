"""libs/plot_utils.py
For Image Classification - Digit Classifier
Author: Adam J. Vogt (Aug. 2017)
----------

utils for plotting data and results

"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from sklearn.metrics import confusion_matrix


def plot_examples(imgs, labels):
    """Plot random example for each digit in labels
    
    Parameters
    ----------
    imgs : array, shape = [n_examples, 784]
        array containing flattened arrays of the digit images
    
    labels : array, shape = [n_examples, ]
        array containing the image labels
        
    Returns
    -------
    None
    
    """
    fig, ax = plt.subplots(nrows=2, 
                           ncols=5, 
                           sharex=True,
                           sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img_subset = (labels==i)
        img_idx = np.random.choice(
            range(labels[img_subset].shape[0]))
        img = imgs[img_subset][img_idx,:].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys')
        ax[i].set_title('Label: %i' 
                        % labels[img_subset][img_idx])
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()
    

def pixel_dist(imgs):
    """Plot distribution of pixel values over all images
    
    Parameters
    ----------
    imgs : array, shape = [n_examples, 784]
        array containing flattened arrays of the digit images
        
    Returns
    -------
    None
    
    """
    
    plt.hist(imgs.ravel(),bins=50,log=True,edgecolor='black')
    plt.xlabel('Pixel Value')
    plt.ylabel('Number of Pixels')
    plt.title('Distribution of Pixel Values')
    plt.show()


def label_dist(labels_train, labels_test):
    """Plot random example for each digit in labels
    
    Parameters
    ----------
    labels_train : array, shape = [n_examples, ]
        array containing the image labels for training set
    
    labels_test : array, shape = [n_examples, ]
        array containing the image labels for test set
        
    Returns
    -------
    None
    
    """
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,4))
    
    # Training Data
    ax1.bar(np.unique(labels_train),np.bincount(labels_train),
            edgecolor='black',color='blue')
    ax1.set_xticks(np.arange(0,10,1))
    ax1.set_ylim([np.bincount(labels_train).mean() - \
                  3*np.bincount(labels_train).std(),
                  np.bincount(labels_train).mean() + \
                  3*np.bincount(labels_train).std()])
    ax1.set_xlabel('Digit')
    ax1.set_ylabel('Number of Examples')
    ax1.set_title('Training Set Distribution')
    
    # Test Data
    ax2.bar(np.unique(labels_test),np.bincount(labels_test),
            edgecolor='black',color='red')
    ax2.set_xticks(np.arange(0,10,1))
    ax2.set_ylim([np.bincount(labels_test).mean() - \
                  3*np.bincount(labels_test).std(),
                  np.bincount(labels_test).mean() + \
                  3*np.bincount(labels_test).std()])
    ax2.set_xlabel('Digit')
    ax2.set_ylabel('Number of Examples')
    ax2.set_title('Test Set Distribution')
    
    plt.show()

    
def plot_statRep(imgs, labels):
    """Plot average and standard devation image for each digit in labels
    
    Parameters
    ----------
    imgs : array, shape = [n_examples, 784]
        array containing flattened arrays of the digit images
    
    labels : array, shape = [n_examples, ]
        array containing the image labels
        
    Returns
    -------
    None
    
    """
    fig = plt.figure(figsize=(12, 4))
    
    # Setting up Subplot Grid
    outer_grid = gridspec.GridSpec(1, 2)
    inner_grid1 = gridspec.GridSpecFromSubplotSpec(2, 5,
        subplot_spec=outer_grid[0], wspace=0.0)
    inner_grid2 = gridspec.GridSpecFromSubplotSpec(2, 5,
        subplot_spec=outer_grid[1], wspace=0.0)
    
    # Digit Average
    for i in range(10):
        img = imgs[labels==i].mean(axis=0).reshape(28, 28)
        ax = plt.Subplot(fig, inner_grid1[i])
        ax.imshow(img, cmap='Greys')
        ax.set_title('Label: %i' 
                        % labels[labels==i][i])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
    
    # Digit Standard Deviation
    for i in range(10):
        img = imgs[labels==i].std(axis=0).reshape(28, 28)
        ax = plt.Subplot(fig, inner_grid2[i])
        ax.imshow(img, cmap='Greys')
        ax.set_title('Label: %i' 
                        % labels[labels==i][i])
        ax.set_xticks([])
        ax.set_yticks([])
        fig.add_subplot(ax)
    
    plt.tight_layout()
    plt.show()
    

def plot_pred_examples(imgs, labels, preds):
    """Plot examples of correctly and incorrectly classified images
    
    Parameters
    ----------
    imgs : array, shape = [n_examples, 784]
        array containing flattened arrays of the digit images
    
    labels : array, shape = [n_examples, ]
        array containing the image labels
    
    pred : array, shape = [n_examples, ]
        array containing the image label predictions
        
    Returns
    -------
    None
    
    """
    
    fig = plt.figure(figsize=(12, 9))
    
    # Setting up Subplot Grid
    outer_grid = gridspec.GridSpec(1, 2)
    inner_grid1 = gridspec.GridSpecFromSubplotSpec(4, 3,
        subplot_spec=outer_grid[0], wspace=0.0)
    inner_grid2 = gridspec.GridSpecFromSubplotSpec(4, 3,
        subplot_spec=outer_grid[1], wspace=0.0)
    
    # Correctly Classified Samples
    for i in range(10):
        img_subset = (preds==labels)&(labels==i)
        img_idx = np.random.choice(
            range(labels[img_subset].shape[0]))
        img = imgs[img_subset][img_idx,:].reshape(28, 28)
        ax = plt.Subplot(fig, inner_grid1[i])
        ax.imshow(img, cmap='Greys')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('T: %i, P: %i' 
                     % (labels[img_subset][img_idx],
                        preds[img_subset][img_idx]))
        fig.add_subplot(ax)
    
    # Misclassified Samples
    for i in range(10):
        img_subset = (preds!=labels)&(labels==i)
        img_idx = np.random.choice(
            range(labels[img_subset].shape[0]))
        img = imgs[img_subset][img_idx,:].reshape(28, 28)
        ax = plt.Subplot(fig, inner_grid2[i])
        ax.imshow(img, cmap='Greys')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('T: %i, P: %i' 
                     % (labels[img_subset][img_idx],
                        preds[img_subset][img_idx]))
        fig.add_subplot(ax)
    plt.show()
    

def plot_confusion_matrix(labels, preds):
    """Plot confusion matrix for predictions
    
    Parameters
    ----------
    labels : array, shape = [n_examples, ]
        array containing the image labels
    
    pred : array, shape = [n_examples, ]
        array containing the image label predictions
        
    Returns
    -------
    None
    
    """
    
    # Calculating confusion matrix
    cm = confusion_matrix(y_true=labels, y_pred=preds)
    
    # Plot
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.7)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i,
                    s=cm[i, j],
                    va='center', ha='center')
    ax.tick_params(axis='both', bottom='off',
                   which='both', labeltop='on')
    ax.xaxis.set_ticks(np.arange(0, cm.shape[0],1))
    ax.xaxis.set_ticks(np.arange(0.5, cm.shape[0],1), minor='True')
    ax.yaxis.set_ticks(range(cm.shape[1]))
    ax.yaxis.set_ticks(np.arange(0.5, cm.shape[1],1), minor='True')
    ax.set_xlabel('Predicted Label')
    ax.xaxis.set_label_position('top')
    ax.set_ylabel('True Label')
    ax.grid(color='black', which='minor')
    plt.show()