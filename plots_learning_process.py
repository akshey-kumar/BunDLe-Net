import sys
sys.path.append(r'../')
import numpy as np
from functions import *

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
X = data.neuron_traces.T
B = data.states
#state_names = data.state_names
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
#plotting_neuronal_behavioural(X, B, state_names=state_names)


### Preprocess and prepare data for BundLe Net
time, X = preprocess_data(X, data.fps)
X_, B_ = prep_data(X, B, win=15)

### Deploy BunDLe Net
model = BunDLeNet(latent_dim=3)
model.build(input_shape=X_.shape)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

for i in range(20):

    loss_array = train_model(X_,
                 B_, 
                 model,
                 optimizer, 
                 gamma=0.9, 
                 n_epochs=50
                             )


    ### Projecting into latent space
    Y0_ = model.tau(X_[:,0]).numpy() 

    algorithm = 'BunDLeNet'
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    plot_ps_(fig, ax, Y0_, B_, state_names = state_names, legend=False)
    plt.savefig('figures/learning_process/plot_learning_process' + str(i) + '.pdf')