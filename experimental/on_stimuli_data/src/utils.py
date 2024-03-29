import sys
sys.path.append(r'../')
import mat73
import numpy as np
import seaborn as sns
from tqdm import tqdm
from scipy import signal
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses

from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

############################################
####### Data preprocessing functions #######
############################################

def bandpass(traces, f_l, f_h, sampling_freq):
    """
    Apply a bandpass filter to the input traces.

    Parameters:
        traces (np.ndarray): Input traces to be filtered.
        f_l (float): Lower cutoff frequency in Hz.
        f_h (float): Upper cutoff frequency in Hz.
        sampling_freq (float): Sampling frequency in Hz.

    Returns:
        filtered (np.ndarray): Filtered traces.

    """    
    cut_off_h = f_h*sampling_freq/2 ## in units of sampling_freq/2
    cut_off_l= f_l*sampling_freq/2 ## in units of sampling_freq/2
    #### Note: the input f_l and f_h are angular frequencies. Hence the argument sampling_freq in the function is redundant: since the signal.butter function takes angular frequencies if fs is None.
    
    sos = signal.butter(4, [cut_off_l, cut_off_h], 'bandpass', fs=sampling_freq, output='sos')
    ### filtering the traces forward and backwards
    filtered = signal.sosfilt(sos, traces)
    filtered = np.flip(filtered, axis=1)
    filtered = signal.sosfilt(sos, filtered)
    filtered = np.flip(filtered, axis=1)
    return filtered

def preprocess_data(X, fps):
    """Preprocess the input data by applying bandpass filtering.
    
    Args:
        X: Input data.
        fps (float): Frames per second.
    
    Returns:
        numpy.ndarray: Preprocessed data after bandpass filtering.
    """
    time = 1 / fps * np.arange(0, X.shape[0])
    filtered = bandpass(X.T, f_l=1e-10, f_h=0.05, sampling_freq=fps).T

    return time, filtered


########################################################
##########  Preparing the data for BunDLe Net ##########
########################################################

def prep_data(X, B, win=15):
    """
    Prepares the data for the BundleNet algorithm by formatting the input neuronal and behavioral traces.

    Parameters:
        X : np.ndarray
            Raw neuronal traces of shape (n, t), where n is the number of neurons and t is the number of time steps.
        B : np.ndarray
            Raw behavioral traces of shape (t,), representing the behavioral data corresponding to the neuronal
            traces.
        win : int, optional
            Length of the window to feed as input to the algorithm. If win > 1, a slice of the time series is used 
            as input.

    Returns:
        X_paired : np.ndarray
            Paired neuronal traces of shape (m, 2, win, n), where m is the number of paired windows,
            2 represents the current and next time steps, win is the length of each window,
            and n is the number of neurons.
        B_1 : np.ndarray
            Behavioral traces corresponding to the next time step, of shape (m,). Each value represents
            the behavioral data corresponding to the next time step in the paired neuronal traces.

    """
    win+=1
    X_win = np.zeros((X.shape[0]-win+1, win, X.shape[1]))
    for i, _ in enumerate(X_win):
        X_win[i] = X[i:i+win]

    Xwin0, Xwin1 = X_win[:,:-1,:], X_win[:,1:,:]
    B_1 = B[win-1:]
    X_paired = np.array([Xwin0, Xwin1])
    X_paired = np.transpose(X_paired, axes=(1,0,2,3))
    
    return X_paired, B_1

def timeseries_train_test_split(X_paired, B_1):
    """
    Perform a train-test split for time series data without shuffling, based on a specific fold.

    Parameters:
        X_paired : np.ndarray
            Paired neuronal traces of shape (m, 2, win, n), where m is the number of paired windows,
            2 represents the current and next time steps, win-1 is the length of each window excluding the last time 
            step,and n is the number of neurons.
        B_1 : np.ndarray
            Behavioral traces corresponding to the next time step, of shape (m,). Each value represents the behavioral 
            data corresponding to the next time step in the paired neuronal traces.

    Returns:
        X_train : np.ndarray
            Training set of paired neuronal traces, of shape (m_train, 2, win, n), where m_train is the number of 
            paired windows in the training set.
        X_test : np.ndarray
            Test set of paired neuronal traces, of shape (m_test, 2, win, n), where m_test is the number of paired 
            windows in the test set.
        B_train_1 : np.ndarray
            Behavioral traces corresponding to the next time step in the training set, of shape (m_train,).
        B_test_1 : np.ndarray
            Behavioral traces corresponding to the next time step in the test set, of shape (m_test,).

    """
    # Train test split 
    kf = KFold(n_splits=7)
    for i, (train_index, test_index) in enumerate(kf.split(X_paired)):
        if i==4: 
            # Train test split based on a fold
            X_train, X_test = X_paired[train_index], X_paired[test_index]
            B_train_1, B_test_1 = B_1[train_index], B_1[test_index]        

            return X_train, X_test, B_train_1, B_test_1


def tf_batch_prep(X_, B_, batch_size = 100):
    """
    Prepare datasets for TensorFlow by creating batches.

    Parameters:
        X_ : np.ndarray
            Input data of shape (n_samples, ...).
        B_ : np.ndarray
            Target data of shape (n_samples, ...).
        batch_size : int, optional
            Size of the batches to be created. Default is 100.

    Returns:
        batch_dataset : tf.data.Dataset
            TensorFlow dataset containing batches of input data and target data.

    This function prepares datasets for TensorFlow by creating batches. It takes input data 'X_' and target data 'B_'
    and creates a TensorFlow dataset from them.

    The function returns the prepared batch dataset, which will be used for training the TensorFlow model.
    """
    batch_dataset = tf.data.Dataset.from_tensor_slices((X_, B_))
    batch_dataset = batch_dataset.batch(batch_size)
    return batch_dataset


####################################################
#################### Evaluation ####################
####################################################

flat_partial = lambda x: x.reshape(x.shape[0],-1)

def r2_single(y_true, y_pred):
    mse = tf.keras.losses.MeanSquaredError()
    return 1 - mse(y_pred, y_true)/tf.math.reduce_variance(y_true)

def r2(Y_true, Y_pred):
    r2_list=[]
    for i in range(Y_true.shape[-1]):
        R2 = r2_single(Y_true[:,i], Y_pred[:,i])
        r2_list.append(R2)
    r2_list = tf.stack(r2_list)
    return tf.math.reduce_mean(r2_list)

def hits_at_rank(rank, Y_test, Y_pred):
    nbrs = NearestNeighbors(n_neighbors=rank, algorithm='ball_tree').fit(Y_test)
    distances, indices = nbrs.kneighbors(Y_test)
    return np.mean(np.linalg.norm(Y_pred - Y_test, axis=1) < distances[:,-1])


########################################
########## Plotting functions ########## 
########################################

def plotting_neuronal_behavioural(X,B, state_names=[], vmin=0, vmax=2):
    fig, axs = plt.subplots(2,1,figsize=(10,4))
    im0 = axs[0].imshow(X.T,aspect='auto', vmin=vmin,vmax=vmax, interpolation='None')
    # tell the colorbar to tick at integers
    cax0 = plt.colorbar(im0)
    axs[0].set_xlabel("time $t$")
    axs[0].set_ylabel("Neuronal activation")
    
    # get discrete colormap
    colors = sns.color_palette('pastel', len(state_names))
    cmap = cm.colors.ListedColormap(colors)
    cmap = plt.get_cmap('Pastel1', np.max(B) - np.min(B) + 1)
    im1 = axs[1].imshow([B], cmap=cmap, vmin=np.min(B) - 0.5, vmax=np.max(B) + 0.5 , aspect='auto')
    # tell the colorbar to tick at integers
    cax = plt.colorbar(im1, ticks=np.arange(np.min(B), np.max(B) + 1))
    if state_names != []:
        cax.ax.set_yticklabels(state_names)
    axs[1].set_xlabel("time $t$")
    axs[1].set_ylabel("Behaviour")
    axs[1].set_yticks([])
    plt.show()


def plot_phase_space(Y, B, state_names, show_points=False, legend=True, **kwargs):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    plot_ps_(fig, ax, Y=Y, B=B, state_names=state_names, show_points=show_points, legend=legend, **kwargs)
    plt.show()
    return fig, ax


def plot_ps_(fig, ax, Y, B, state_names, show_points=False, legend=True, colors=None, **kwargs):
    if colors is None:
        colors = sns.color_palette('deep', len(state_names))
        color_dict = {name: color for name, color in zip(np.unique(B), colors)}

    for i in range(len(Y) - 1):
        d = (Y[i+1] - Y[i])
        ax.quiver(Y[i, 0], Y[i, 1], Y[i, 2],
                  d[0], d[1], d[2],
                  color=color_dict[B[i]], arrow_length_ratio=0.1/np.linalg.norm(d), linewidths=1, **kwargs)
    ax.set_axis_off()  

    if legend == True:
        # Create legend
        legend_elements = [Line2D([0], [0], color=color_dict[b], lw=4, label=state_names[b]) for b in color_dict]
        ax.legend(handles=legend_elements)

    if show_points == True:
        ax.scatter(Y[:,0], Y[:,1], Y[:,2], c='k', s=1, cmap = ListedColormap(colors))
    return fig, ax


def rotating_plot(Y, B, state_names, show_points=False, legend=True, filename='rotation.gif', **kwargs):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')

    def rotate(angle):
        ax.view_init(azim=angle)

    fig, ax = plot_ps_(fig, ax, Y=Y, B=B, state_names=state_names, show_points=show_points, legend=legend, **kwargs)
    rot_animation = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 362, 5), interval=150)
    rot_animation.save(filename, dpi=150, writer='imagemagick')
    plt.show()
    return ax


def plot_latent_timeseries(Y, B, state_names):
    plt.figure(figsize=(19,5))
    cmap = plt.get_cmap('Pastel1', np.max(B) - np.min(B) + 1)
    im = plt.imshow([B],aspect=600,cmap=cmap, vmin=np.min(B) - 0.5, vmax=np.max(B) + 0.5)
    #cbar = plt.colorbar(ticks=np.arange(len(state_names)))
    cbar = plt.colorbar(im, ticks=np.arange(np.min(B), np.max(B) + 1))
    cbar.ax.set_yticklabels(state_names) 
    plt.plot(Y/np.max(np.abs(Y))/3)
    plt.xlabel("time $t$")
    plt.axis([0,Y.shape[0],-0.5,0.5])
    plt.show()
