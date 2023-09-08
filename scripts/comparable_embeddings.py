import sys
sys.path.append(r'../')
import numpy as np
from functions import *

import os
os.chdir('..')

'''
Layers of the pretrained model on worm_num_i are  
used in learning a comparable embedding on worm_no_j.
so data corresponds to worm_no_j
model_old corresponds to worm_no_i
model_new will be trained on worm_no_j data
'''

worm_num_i = 0
# Load old model created in new_runs.py (of worm_no_i)
## Load old data (reqiured to build old model)
b_neurons = [
    'AVAR',
    'AVAL',
    'SMDVR',
    'SMDVL',
    'SMDDR',
    'SMDDL',
    'RIBR',
    'RIBL',]
data = Database(data_set_no=worm_num_i)
data.exclude_neurons(b_neurons)
X = data.neuron_traces.T
B = data.states
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
time, X = preprocess_data(X, data.fps)
X_, B_ = prep_data(X, B, win=15)
## Get weights of old model
model_old = BunDLeNet(latent_dim=3)
model_old.build(input_shape=X_.shape)
model_old.load_weights('data/generated/BunDLeNet_model_worm_' + str(worm_num_i))
weights_T_Y = model_old.T_Y.get_weights()
weights_predictor = model_old.predictor.get_weights()


worm_num_j = 2
# Load new data and new model (of worm_no_j)
for worm_num_j in range(5):
    data = Database(data_set_no=worm_num_j)
    data.exclude_neurons(b_neurons)
    X = data.neuron_traces.T
    B = data.states
    time, X = preprocess_data(X, data.fps)
    X_, B_ = prep_data(X, B, win=15)

    ### Initialise new model with old T_Y and predictor
    model_new = BunDLeNet(latent_dim=3)
    model_new.build(input_shape=X_.shape)
    model_new.T_Y.set_weights(weights_T_Y)
    model_new.predictor.set_weights(weights_predictor)
    ### Freeze weights of predictor and T_Y
    model_new.T_Y.trainable = False
    model_new.predictor.trainable = False
    optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

    loss_array = train_model(X_,
                 B_,
                 model_new,
                 optimizer,
                 gamma=0.9,
                 n_epochs=2000,
                 pca_init=False
                             )

    ### Projecting into latent space
    Y0_ = model_new.tau(X_[:,0]).numpy() 
    #plot_phase_space(Y0_, B_, state_names = state_names)
    
    algorithm = 'BunDLeNet'
    np.savetxt('data/generated/saved_Y/comparable_embeddings/Y0__' + algorithm + '_worm_' + str(worm_num_j), Y0_)
    np.savetxt('data/generated/saved_Y/comparable_embeddings/B__' + algorithm + '_worm_' + str(worm_num_j), B_)
    
