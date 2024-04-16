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
        self.tau_s = self._build_tau_network()
        self.tau_i = self._build_tau_network()
        self.tau_m = self._build_tau_network()
        self.post_tau = tf.keras.Sequential([
            layers.Concatenate(axis=1),
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
            layers.Dense(20, activation='relu'),
            layers.Dense(10, activation='relu'),
            layers.Dense(7, activation='relu'),
            layers.Dense(3, activation='relu'),
            layers.Dense(1, activation='linear'),
        ])


    def call(self, inputs):
        Xs, Xi, Xm = inputs
        # Upper arm of commutativity diagram
        Yt1_upper_s = self.tau_s(Xs[:,1])
        Yt1_upper_i = self.tau_i(Xi[:,1])
        Yt1_upper_m = self.tau_m(Xm[:,1])
        Yt1_upper = self.post_tau([Yt1_upper_s, Yt1_upper_i, Yt1_upper_m])
        Bt1_upper = self.predictor(Yt1_upper) 

        # Lower arm of commutativity diagram
        Yt1_lower_s = self.tau_s(Xs[:,0])
        Yt1_lower_i = self.tau_i(Xi[:,0])
        Yt1_lower_m = self.tau_m(Xm[:,0])
        Yt_lower = self.post_tau([Yt1_lower_s, Yt1_lower_i, Yt1_lower_m])
        Yt1_lower = Yt_lower + self.T_Y(Yt_lower)

        return Yt1_upper, Yt1_lower, Bt1_upper



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
                  color=colors[B[i]], arrow_length_ratio=0.0001/np.linalg.norm(d), linewidths=1, **kwargs)
    #ax.set_axis_off()
    ax.set_xlabel('sensory neurons axis ')
    ax.set_ylabel('inter neuron axis')
    ax.set_zlabel('motor neuron axis')

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


#from axis_decomp import *

### Load Data (and excluding behavioural neurons)
worm_num = 3

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
Xs_ = X_[:,:,:, mask==1]
Xi_ = X_[:,:,:, mask==2]
Xm_ = X_[:,:,:, mask==3]

### Deploy BunDLe Net
model = BunDLeNet(latent_dim=3)
model.build([Xs_.shape, Xi_.shape, Xm_.shape])
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# PCA init and best of 5 init are not yet ipmlemented for this framework
# of multiple embedders for different neuronal populations. Hence always set
# them to false.
loss_array = train_model((Xs_, Xi_, Xm_),
             B_,
             model,
             optimizer,
             gamma=0.9, 
             n_epochs=3000,
             pca_init=False,
             best_of_5_init=False
                         )

# Training losses vs epochs
'''
plt.figure()
for i, label in enumerate(["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$","Total loss $\mathcal{L}$" ]):
    plt.semilogy(loss_array[:,i], label=label)

plt.legend()
plt.show()
'''
### Projecting into latent space
Y0s_ = model.tau_s(Xs_[:,0])
Y0i_ = model.tau_i(Xi_[:,0])
Y0m_ = model.tau_m(Xm_[:,0])
Y0_ = model.post_tau([Y0s_, Y0i_, Y0m_]).numpy() 

model.post_tau.get_weights()

algorithm = 'BunDLeNet'
# Save the weights
# model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
# np.savetxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num), Y0_)
# np.savetxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num), B_)
# Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
# B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)


### Plotting latent space dynamics
#plot_latent_timeseries(Y0_, B_, state_names)
#plot_phase_space(Y0_, B_, state_names = state_names)
rotating_plot(Y0_, B_,filename='figures/rotation_axis_decomp/rotation'+ algorithm + '_worm_'+str(worm_num) +'.gif', state_names=state_names, legend=False)

