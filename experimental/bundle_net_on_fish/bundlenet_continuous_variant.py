import sys
sys.path.append(r'../')
import numpy as np
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.losses import Loss
from tensorflow.keras.models import Model
from tensorflow.keras import layers, losses
from functions import tf_batch_prep

################################################################################
##### BunDLe Net --- Architecture and functions for training - continuous B ####
################################################################################
class BunDLeNet(Model):
    """Behaviour and Dynamical Learning Network (BunDLeNet) model.
    
    This model represents the BunDLe Net's architecture for deep learning and is based on the commutativity
    diagrams. The resulting model is dynamically consistent (DC) and behaviourally consistent (BC) as per 
    the notion described in the paper.
    
    Args:
        latent_dim (int): Dimension of the latent space.
    """
    def __init__(self, latent_dim, num_behaviour):
        super(BunDLeNet, self).__init__()
        self.latent_dim = latent_dim
        self.num_behaviour = num_behaviour
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
            layers.Dense(num_behaviour, activation='linear')
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

    DCC_loss = mse(yt1_upper, yt1_lower)
    behaviour_loss = mse(b_train_1, bt1_upper)
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
    

def pca_initialisation(X_, tau, latent_dim):
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
                     n_epochs=20,
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

def train_model(X_train, B_train_1, model, optimizer, gamma, n_epochs, pca_init=False, best_of_5_init=False):
    """Training BunDLe Net
    
    Args:
        X_train: Training input data.
        B_train_1: Training output data.
        model: Instance of the BunDLeNet class.
        optimizer: Optimizer for model training.
        gamma (float): Weight for the DCC loss component.
        n_epochs (int): Number of training epochs.
        pca_initialisation (bool)
    
    Returns:
        numpy.ndarray: Array of loss values during training.
    """
    train_dataset = tf_batch_prep(X_train, B_train_1)
    
    if pca_init:
        pca_initialisation(X_train, model.tau, model.latent_dim)
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
