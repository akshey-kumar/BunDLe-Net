import sys
sys.path.append('../..')
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, losses
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from bundlenet_continuous_variant import BunDLeNet, train_model
from functions import preprocess_data, prep_data, plotting_neuronal_behavioural, plot_latent_timeseries, timeseries_train_test_split
from plotting_functions import plot_phase_space
from ray import tune, train
from ray.tune.search.hyperopt import HyperOptSearch

import os


fish_data_id = '220127_F4_F4_run2' #220119_F2_F2_run11, 220127_F4_F4_run2
path_neuronal_data ='../../data/raw/fish_cilia/traces_with_vigour_directionality_behaviours/' + fish_data_id +'_cells_spike_rate_signals.npy' 
path_behaviour_data1 ='../../data/raw/fish_cilia/traces_with_vigour_directionality_behaviours/' + fish_data_id +'_directionality.npy'
path_behaviour_data2 ='../../data/raw/fish_cilia/traces_with_vigour_directionality_behaviours/' + fish_data_id +'_vigour.npy'

# Loading and preparing data for BunDLe-Net
X = np.load(path_neuronal_data).T
B1 = np.load(path_behaviour_data1)
B2 = np.load(path_behaviour_data2)
B = np.c_[B1, B2]

# Remove NaNs
B = B[~np.isnan(X[:,0])]
X = X[~np.isnan(X[:,0])]

# Scaling
B = StandardScaler(with_mean=False).fit_transform(B)
X = StandardScaler(with_mean=False).fit_transform(X)

# Set the parameters to be tuned here
search_space = {
    "win": tune.grid_search([2]),
    "T_Y_option": tune.grid_search(['linear', 'non-linear']),
    'gamma': tune.grid_search([0.9]),
    'latent_dim': tune.grid_search([2,3,4,5,6,7,8,9,10])
} 


def objective(config):

    models = []
    history = []

    X_, B_ = prep_data(X, B, win=config["win"])
    X_train, X_test, B_train_1, B_test_1 = timeseries_train_test_split(X_, B_)

    for i in range(5):
        T_Y_options  = {
            'linear': tf.keras.Sequential([
                layers.Dense(config['latent_dim'], activation='linear'),
                layers.Normalization(axis=-1),
            ]),
            'non-linear': tf.keras.Sequential([
                layers.Dense(config['latent_dim'], activation='relu'),
                layers.Dense(2*config['latent_dim'], activation='relu'),
                layers.Dense(config['latent_dim'], activation='linear'),
                layers.Normalization(axis=-1),
            ])
        }
        ### Deploy BunDLe Net
        model = BunDLeNet(latent_dim=config['latent_dim'], num_behaviour=B_.shape[1])
        model.build(input_shape=X_.shape)
        optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
        model.T_Y = T_Y_options[config["T_Y_option"]]

        train_history, test_history = train_model(
            X_train,
            B_train_1,
            model,
            optimizer,
            gamma=config['gamma'], 
            n_epochs=200,
            pca_init=False,
            best_of_5_init=False,
            validation_data = (X_test, B_test_1)
        )       
        models.append(model)
        history.append([train_history, test_history])

    # Choosing best model based on the test loss
    history = np.array(history)
    train_loss = history[:,0,-1,-1]
    test_loss = history[:,1,-1,-1]

    idx_optimal = np.argmin(test_loss)
    model_opt = models[idx_optimal]
    train_loss_opt = train_loss[idx_optimal]
    test_loss_opt = test_loss[idx_optimal]
    train.report({"test_loss":test_loss_opt, "train_loss":train_loss_opt})


#algo = HyperOptSearch()
tuner = tune.Tuner(
    objective,
    tune_config=tune.TuneConfig(
        num_samples=5,
        metric="test_loss",
        mode="min",
        #search_alg=algo,
    ),
    param_space=search_space,
)
results = tuner.fit()
results.get_best_result(metric="test_loss", mode="min").config
print(results)