import sys
sys.path.append('../..')
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from bundlenet_continuous_variant import BunDLeNet, train_model
from functions import preprocess_data, prep_data, plotting_neuronal_behavioural, plot_latent_timeseries, timeseries_train_test_split
from plotting_functions import plot_phase_space

import os
os.chdir('../..')

#path_neuronal_data ='data/raw/fish_cilia/traces_with_vigour_directionality_behaviours/220210_F2_F2_run5_cells_fluorescence_signals.npy' 
path_neuronal_data ='data/raw/fish_cilia/traces_with_vigour_directionality_behaviours/220210_F2_F2_run5_cells_spike_rate_signals.npy' 
path_behaviour_data1 ='data/raw/fish_cilia/traces_with_vigour_directionality_behaviours/220210_F2_F2_run5_directionality.npy'
path_behaviour_data2 ='data/raw/fish_cilia/traces_with_vigour_directionality_behaviours/220210_F2_F2_run5_vigour.npy'

X = np.load(path_neuronal_data).T
B1 = np.load(path_behaviour_data1)
B2 = np.load(path_behaviour_data2)
B = np.c_[B1, B2]

### Remove NaNs
B = B[~np.isnan(X[:,0])]
X = X[~np.isnan(X[:,0])]
B.shape, X.shape

### Scaling
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
B = StandardScaler(with_mean=False).fit_transform(B)
X = StandardScaler(with_mean=False).fit_transform(X)

algorithm = 'BunDLeNet'
X_, B_ = prep_data(X, B, win=1)
X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)

### Deploy BunDLe Net
model = BunDLeNet(latent_dim=3, num_behaviour=B_.shape[1])
model.build(input_shape=X_.shape)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)


train_history, test_history = train_model(
    X_train,
    B_train_1,
    model,
    optimizer,
    gamma=0.9, 
    n_epochs=500,
    pca_init=False,
    best_of_5_init=False, 
    validation_data = (X_test, B_test_1)
)


# Training losses vs epochs
plt.figure()
for i, label in  enumerate(["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$","Total loss $\mathcal{L}$" ]):
    plt.semilogy(train_history[:,i], label=label)

plt.semilogy(test_history[:,-1], label="Test loss", linestyle="--", color='g')   
plt.legend()
plt.show()

"""
train_history = train_model(
    X_train,
    B_train_1,
    model,
    optimizer,
    gamma=0.9, 
    n_epochs=5,
    pca_init=False,
    best_of_5_init=False, 
)

# Training losses vs epochs
plt.figure()
for i, label in  enumerate(["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$","Total loss $\mathcal{L}$" ]):
    plt.semilogy(train_history[:,i], label=label)
plt.legend()
plt.show()
"""
