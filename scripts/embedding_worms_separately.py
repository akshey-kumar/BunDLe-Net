import sys
sys.path.append(r'../')
import numpy as np
from functions import *

import os
os.chdir('..')

### Load Data (and excluding behavioural neurons)
for worm_num in range(5):
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

    ### Preprocess and prepare data for BundLe Net
    time, X = preprocess_data(X, data.fps)
    X_, B_ = prep_data(X, B, win=15)

    ### Deploy BunDLe Net
    model = BunDLeNet(latent_dim=3)
    model.build(input_shape=X_.shape)
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    #X_train, X_test, B_train, B_test = timeseries_train_test_split(X_, B_)
    loss_array = train_model(X_,
                 B_,
                 model,
                 optimizer,
                 gamma=0.9, 
                 n_epochs=3000,
                 pca_init=False
                             )
    ### Projecting into latent space
    Y0_ = model.tau(X_[:,0]).numpy() 

    algorithm = 'BunDLeNet'
    # Save the weights
    model.save_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num))
    np.savetxt('data/generated/saved_Y/new_runs_3000_epochs/Y0__' + algorithm + '_worm_' + str(worm_num), Y0_)
    np.savetxt('data/generated/saved_Y/new_runs_3000_epochs/B__' + algorithm + '_worm_' + str(worm_num), B_)
    
    #plot_phase_space(Y0_, B_, state_names = state_names)

