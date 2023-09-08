import sys
sys.path.append(r'../..')
import numpy as np
from functions import *

import os
os.chdir('../..')

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
        self.tau = self._build_tau_network()
        self.post_tau = tf.keras.Sequential([
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


    def _build_tau_network(self):
        return tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(50, activation='relu'),
            layers.Dense(30, activation='relu'),
            layers.Dense(25, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(self.latent_dim, activation='linear')
        ])

    def call(self, X):
        # Upper arm of commutativity diagram
        Yt1_upper = self.tau(X[:,1])
        Yt1_upper = self.post_tau(Yt1_upper)
        Bt1_upper = self.predictor(Yt1_upper) 

        # Lower arm of commutativity diagram
        Yt1_lower = self.tau(X[:,0])
        Yt_lower = self.post_tau(Yt1_lower)
        Yt1_lower = Yt_lower + self.T_Y(Yt_lower)

        return Yt1_upper, Yt1_lower, Bt1_upper


### Load Data (and excluding behavioural neurons)
worm_num = 0
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
mask = data.categorise_neurons()
X = data.neuron_traces.T
B = data.states
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']

### Preprocess and prepare data for BundLe Net
time, X = preprocess_data(X, data.fps)
X_, B_ = prep_data(X, B, win=15)

### Deploy BunDLe Net
model = BunDLeNet(latent_dim=3)
model.build(input_shape=X_.shape)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

loss_array = train_model(X_,
             B_,
             model,
             optimizer,
             gamma=0.9, 
             n_epochs=1,
             pca_init=False,
             best_of_5_init=False
                         )

# Training losses vs epochs
plt.figure()
for i, label in  enumerate(["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$","Total loss $\mathcal{L}$" ]):
    plt.semilogy(loss_array[:,i], label=label)
plt.legend()
plt.show()

### Projecting into latent space
Y0_ = model.tau(X_[:,0])
Y0_ = model.post_tau(Y0_).numpy() 


model.post_tau.get_weights()

algorithm = 'BunDLeNet'
# Save the weights
# model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
# np.savetxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num), Y0_)
# np.savetxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num), B_)
# Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
# B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)

### Plotting latent space dynamics
plot_latent_timeseries(Y0_, B_, state_names)
plot_phase_space(Y0_, B_, state_names = state_names)

