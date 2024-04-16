import sys
sys.path.append(r'../')
import numpy as np
from functions import *

### Load Data (and excluding behavioural neurons)
worm_num = 4
b_neurons = [
    'AVAR',
    'AVAL',
    'SMDVR',
    'SMDVL',
    'SMDDR',
    'SMDDL',
    'RIBR',
    'RIBL',]
data = Database(data_set_no=worm_num)
data.exclude_neurons(b_neurons)
X = data.neuron_traces.T
B = data.states

X_, B_ = prep_data(X, B, win=15)


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
            layers.Normalization(axis=-1)
        ])
        self.predictor = tf.keras.Sequential([
            layers.Dense(latent_dim, activation='linear'),
            layers.Dense(10, activation='relu'),
            layers.Dense(25, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dense(50, activation='relu'),
            layers.Dense(1815,activation='linear'),
            layers.Reshape((15, 121))
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
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
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
    

def train_model(X_train, B_train_1, model, optimizer, gamma, n_epochs, pca_init=False):
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


### Deploy BunDLe Net
model = BunDLeNet(latent_dim=3)
model.build(input_shape=X_.shape)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#X_train, X_test, B_train, B_test = timeseries_train_test_split(X_, B_)
loss_array = train_model(
                         X_,
                         X_[:,1], 
                         model,
                         optimizer,
                         gamma=0.2, 
                         n_epochs=500, 
                         pca_init=True)

# Training losses vs epochs
plt.figure()
for i, label in  enumerate(["DCC_loss", "behaviour_loss","total_loss" ]):
    plt.plot(loss_array[:,i], label=label)
plt.legend()
#plt.ylim(0,.01)


### Projecting into latent space
Y0_ = model.tau(X_[:,0]).numpy() 

algorithm = 'BunDLeNet'
#np.savetxt('Saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num), Y0_)
#np.savetxt('Saved_Y/B__' + algorithm + '_worm_' + str(worm_num), B_)
#Y0_ = np.loadtxt('Saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
#B_ = np.loadtxt('Saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)

### Plotting latent space dynamics
plt.figure(figsize=(19,5))
plt.imshow([B_],aspect=600,cmap="Pastel1")
cbar = plt.colorbar(ticks=np.arange(8))
cbar.ax.set_yticklabels(['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reverse', 'Slowing', 'Ventral turn']) 
plt.plot(Y0_/Y0_.max()/3)
plt.xlabel("time $t$")
plt.axis([0,Y0_.shape[0],-0.5,0.5])
plt.show()

plot_phase_space(Y0_, B_, show_points=True)
plt.show()
### Run to produce rotating 3-D plot
#rotating_plot(Y0_, B_,filename='rotation_'+ algorithm + '_worm_'+str(worm_num) +'.gif')

### Performing PCA on the latent dimension (to check if there are redundant or correlated components)
pca = PCA()
Y_pca = pca.fit_transform(Y0_)
plt.figure(figsize=(19,5))
plt.imshow([B_],aspect=600,cmap="Pastel1")
cbar = plt.colorbar(ticks=np.arange(8))
cbar.ax.set_yticklabels(['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reverse', 'Slowing', 'Ventral turn']) 
plt.plot(Y_pca/Y_pca.max()/3)
plt.xlabel("time $t$")
plt.ylabel("$Y_{pca}$")
plt.axis([0,Y_pca.shape[0],-0.5,0.5])
plt.show()

plt.figure(figsize=(15,3))
plt.imshow([B_],aspect="auto",cmap="Pastel1")
plt.plot(Y_pca[:,2]/3/np.max(np.abs(Y_pca[:,2])), color = 'green')


