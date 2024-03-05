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

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#########################################
############## Loading Data #############
#########################################


class Database:
    """
    Loading neuronal and behavioural data from matlab files 

    Attributes:
        data_set_no (int): The number of the data set.
        states (numpy.ndarray): A single array of states, where each number corresponds to a behaviour.
        state_names (list): List of state names.
        neuron_traces (numpy.ndarray): Array of neuron traces.
        neuron_names (numpy.ndarray): Array of neuron names.
        fps (float): Frames per second.

    Methods:
        exclude_neurons: Excludes specified neurons from the database.
        categorise_neurons: Categorises neurons based on whether it is sensory,
                            inter or motor neuron. 

    """
    def __init__(self, data_set_no):
        self.data_set_no = data_set_no
        data_dict = mat73.loadmat('data/raw/NoStim_Data.mat')
        data  = data_dict['NoStim_Data']

        deltaFOverF_bc = data['deltaFOverF_bc'][self.data_set_no]
        derivatives = data['derivs'][self.data_set_no]
        NeuronNames = data['NeuronNames'][self.data_set_no]
        fps = data['fps'][self.data_set_no]
        States = data['States'][self.data_set_no]

        self.states = np.sum([n*States[s] for n, s in enumerate(States)], axis = 0).astype(int) # making a single states array in which each number corresponds to a behaviour
        self.state_names = [*States.keys()]
        self.neuron_traces = np.array(deltaFOverF_bc).T
        #self.derivative_traces = derivatives['traces'].T
        self.neuron_names = np.array(NeuronNames, dtype=object)
        self.fps = fps

        ### To handle bug in dataset 3 where in neuron_names the last entry is a list. we replace the list with the contents of the list
        self.neuron_names = np.array([x if not isinstance(x, list) else x[0] for x in self.neuron_names])


    def exclude_neurons(self, exclude_neurons):
        """
        Excludes specified neurons from the database.

        Args:
            exclude_neurons (list): List of neuron names to exclude.

        Returns:
            None

        """
        neuron_names = self.neuron_names
        mask = np.zeros_like(self.neuron_names, dtype='bool')
        for exclude_neuron in exclude_neurons:
            mask = np.logical_or(mask, neuron_names==exclude_neuron)
        mask = ~mask
        self.neuron_traces = self.neuron_traces[mask] 
        #self.derivative_traces = self.derivative_traces[mask] 
        self.neuron_names = self.neuron_names[mask]

    def _only_identified_neurons(self):
        mask = np.logical_not([x.isnumeric() for x in self.neuron_names])
        self.neuron_traces = self.neuron_traces[mask] 
        #self.derivative_traces = self.derivative_traces[mask] 
        self.neuron_names = self.neuron_names[mask]

    def categorise_neurons(self):
        self._only_identified_neurons()
        neuron_list = mat73.loadmat('data/raw/Order279.mat')['Order279']
        neuron_category = mat73.loadmat('data/raw/ClassIDs_279.mat')['ClassIDs_279']
        category_dict = {neuron: int(category) for neuron, category in zip(neuron_list, neuron_category)}

        mask = np.array([category_dict[neuron] for neuron in self.neuron_names])
        mask_s = mask == 1
        mask_i = mask == 2
        mask_m = mask == 3

        self.neuron_names_s = self.neuron_names[mask_s]
        self.neuron_names_i = self.neuron_names[mask_i]
        self.neuron_names_m = self.neuron_names[mask_m]

        self.neuron_traces_s = self.neuron_traces[mask_s]
        self.neuron_traces_i = self.neuron_traces[mask_i]
        self.neuron_traces_m = self.neuron_traces[mask_m]

        return mask

flat_partial = lambda x: x.reshape(x.shape[0],-1)

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


#################################################################
##### BunDLe Net --- Architecture and functions for training ####
#################################################################

class BunDLeNet(Model):
    """Behaviour and Dynamical Learning Network (BunDLeNet) model.
    
    This model represents the BunDLe Net's architecture for deep learning and is based on the commutativity
    diagrams. The resulting model is dynamically consistent (DC) and behaviourally consistent (BC) as per 
    the notion described in the paper. 
    
    Args:
        latent_dim (int): Dimension of the latent space.
    """
    def __init__(self, latent_dim):
        super(BunDLeNet, self).__init__()
        self.latent_dim = latent_dim
        self.tau = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dense(25, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(latent_dim, activation='linear'),
            layers.Normalization(axis=-1), 
            layers.GaussianNoise(0.05)
        ])
        self.T_Y = tf.keras.Sequential([
            layers.Dense(latent_dim, activation='linear'),
            layers.Normalization(axis=-1),    
        ])
        self.predictor = tf.keras.Sequential([
            layers.Dense(8, activation='linear')
        ]) 

    def call(self, X):
        # Upper arm of commutativity diagram
        Yt1_upper = self.tau(X[:,1])
        Bt1_upper = self.predictor(Yt1_upper) 

        # Lower arm of commutativity diagram
        Yt_lower = self.tau(X[:,0])
        Yt1_lower = Yt_lower + self.T_Y(Yt_lower)

        return Yt1_upper, Yt1_lower, Bt1_upper


def bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_train_1, gamma):
    """Calculate the loss for the BunDLe Net
    
    Args:
        yt1_upper: Output from the upper arm of the BunDLe Net.
        yt1_lower: Output from the lower arm of the BunDLe Net.
        bt1_upper: Predicted output from the upper arm of the BunDLe Net.
        b_train_1: True output for training.
        gamma (float): Tunable weight for the DCC loss component.
    
    Returns:
        tuple: A tuple containing the DCC loss, behavior loss, and total loss.
    """
    mse = tf.keras.losses.MeanSquaredError()
    scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    DCC_loss = mse(yt1_upper, yt1_lower)
    behaviour_loss = scce(b_train_1, bt1_upper)
    total_loss = gamma*DCC_loss + (1-gamma)*behaviour_loss
    return gamma*DCC_loss, (1-gamma)*behaviour_loss, total_loss


class BunDLeTrainer:
    """Trainer for the BunDLe Net model.
    
    This class handles the training process for the BunDLe Net model.
    
    Args:
        model: Instance of the BunDLeNet class.
        optimizer: Optimizer for model training.
    """
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
    
    @tf.function
    def train_step(self, x_train, b_train_1, gamma):
        with tf.GradientTape() as tape:
            yt1_upper, yt1_lower, bt1_upper = self.model(x_train, training=True)
            DCC_loss, behaviour_loss, total_loss = bccdcc_loss(yt1_upper, yt1_lower, bt1_upper, b_train_1, gamma)
        grads = tape.gradient(total_loss, self.model.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        return DCC_loss, behaviour_loss, total_loss
    

def train_model(X_train, B_train_1, model, optimizer, gamma, n_epochs, pca_init=False, best_of_5_init=False):
    """
    Training BunDLe Net
    
    Args:
        X_train: Training input data.
        B_train_1: Training output data.
        model: Instance of the BunDLeNet class.
        optimizer: Optimizer for model training.
        gamma (float): Weight for the DCC loss component.
        n_epochs (int): Number of training epochs.
        pca_initialisation (bool)
        best_of_5_init (bool)
    
    Returns:
        numpy.ndarray: Array of loss values during training.
    """
    train_dataset = tf_batch_prep(X_train, B_train_1)
    if pca_init:
        _pca_initialisation(X_train, model.tau, model.latent_dim)
        model.tau.load_weights('data/generated/tau_pca_weights.h5')

    if best_of_5_init:
        model = _best_of_5_runs(X_train, B_train_1, model, optimizer, gamma)
       
    trainer = BunDLeTrainer(model, optimizer)
    loss_array = np.zeros((1,3))
    epochs = tqdm(np.arange(n_epochs))
    for epoch in epochs:
        for step, (x_train, b_train_1) in enumerate(train_dataset):
            DCC_loss, behaviour_loss, total_loss = trainer.train_step(x_train, b_train_1, gamma=gamma)
            loss_array = np.append(loss_array, [[DCC_loss, behaviour_loss, total_loss]], axis=0)
        epochs.set_description("Losses %f %f %f" %(DCC_loss.numpy(), behaviour_loss.numpy(), total_loss.numpy()))
    loss_array = np.delete(loss_array, 0, axis=0)
    loss_array = loss_array.reshape(n_epochs, int(loss_array.shape[0]//n_epochs), loss_array.shape[-1]).mean(axis=1)
    return loss_array


def _pca_initialisation(X_, tau, latent_dim):
    """
    Initialises BunDLe Net's tau such that its output is the PCA of the input traces.
    PCA initialisation may make the embeddings more reproduceable across runs.
    This function is called within the train_model() function and saves the learned tau weights
    in a .h5 file in the same repository.

    Parameters:
        X_ (np.ndarray): Input data.
        tau (object): BunDLe Net tau (tf sequential layer).
        latent_dim (int): Dimension of the latent space.
    

    """
    ### Performing PCA on the time slice
    X0_ = X_[:,0,:,:]
    X_pca = X_.reshape(X_.shape[0],2,1,-1)[:,0,0,:]
    pca = PCA(n_components = latent_dim, whiten=True)
    pca.fit(X_pca)
    Y0_ = pca.transform(X_pca)
    ### Training tau to reproduce the PCA
    class PCA_encoder(Model):
      def __init__(self, latent_dim):
        super(PCA_encoder, self).__init__()
        self.latent_dim = latent_dim   
        self.encoder = tau
      def call(self, x):
        encoded = self.encoder(x)
        return encoded

    pcaencoder = PCA_encoder(latent_dim = latent_dim)
    opt = tf.keras.optimizers.legacy.Adam(learning_rate=0.01)
    pcaencoder.compile(optimizer=opt,
                  loss='mse',
                  metrics=['mse'])
    history = pcaencoder.fit(X0_,
                          Y0_,
                          epochs=10,
                          batch_size=100,
                          verbose=0,
                          )
    Y0_pred = pcaencoder(X0_).numpy()
    ### Saving weights of this model
    pcaencoder.encoder.save_weights('data/generated/tau_pca_weights.h5')


def _best_of_5_runs(X_train, B_train_1, model, optimizer, gamma):
    """
    Initialises BunDLe net with the best of 5 runs

    Performs 200 epochs of training for 5 random model initialisations 
    and picks the model with the lowest loss
    """
    model_loss = []
    for i in range(5):
        model_ = keras.models.clone_model(model)
        model_.build(input_shape=X_train.shape)
        loss_array = train_model(X_train,
                     B_train_1,
                     model_,
                     optimizer,
                     gamma=gamma, 
                     n_epochs=200,
                     pca_init=False,
                     best_of_5_init=False
                                 )
        model_.save_weights('data/generated/best_of_5_runs_models/model_' + str(i))
        model_loss.append(loss_array[-1,2])

    for n, i in enumerate(model_loss):
        print('model:', n, 'loss:', i)

    ### Load model with least loss
    model.load_weights('data/generated/best_of_5_runs_models/model_' + str(np.argmin(model_loss)))
    return model


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

    for i in range(len(Y) - 1):
        d = (Y[i+1] - Y[i])
        ax.quiver(Y[i, 0], Y[i, 1], Y[i, 2],
                  d[0], d[1], d[2],
                  color=colors[B[i]], arrow_length_ratio=0.1/np.linalg.norm(d), linewidths=1, **kwargs)
    ax.set_axis_off()  

    if legend == True:
        # Create legend
        legend_elements = [Line2D([0], [0], color=c, lw=4, label=state) for c, state in zip(colors, state_names)]
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
