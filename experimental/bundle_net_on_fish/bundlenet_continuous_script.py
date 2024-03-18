import sys
sys.path.append('../..')
import argparse
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from bundlenet_continuous_variant import BunDLeNet, train_model
from .functions import preprocess_data, prep_data, plotting_neuronal_behavioural, plot_latent_timeseries


import os
os.chdir('../..')

parser = argparse.ArgumentParser(description='Run bundle net continuous variant on neuronal and beahvarioual data')
parser.add_argument('--neuronal-data', 
	help='path to neuronal data', 
	default='data/raw/fish_cilia/traces_with_vigour_directionality_behaviours/220210_F2_F2_run5_cells_fluorescence_signals.npy', 
	required=False)
parser.add_argument('--behaviour-data', 
	help='path to behaviour data', 
	default='data/raw/fish_cilia/traces_with_vigour_directionality_behaviours/directionality.npy', 
	required=False)

#parser.add_argument('--neuronal-data', help='path to neuronal data', default='data/raw/fish_cilia/aligned_akshey/X_F2.npy', required=False)
#parser.add_argument('--behaviour-data', help='path to behaviour data', default='data/raw/fish_cilia/aligned_akshey/B_F2.npy', required=False)


if __name__ == '__main__':

	args = parser.parse_args()

	#### Loading and preparing data for BunDLe-Net
	X = np.load(args.neuronal_data).T
	B = np.load(args.behaviour_data)
	print(X.shape)
	print(B.shape)

	algorithm = 'BunDLeNet'
	X_, B_ = prep_data(X, B, win=20)

	### Deploy BunDLe Net
	model = BunDLeNet(latent_dim=3, num_behaviour=1)
	model.build(input_shape=X_.shape)
	optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

	#X_train, X_test, B_train, B_test = timeseries_train_test_split(X_, B_)
	train_history = train_model(
	    X_,
	    B_,
	    model,
	    optimizer,
	    gamma=0.9, 
	    n_epochs=200,
	    pca_init=False
    )

	# Training losses vs epochs
	plt.figure()

	for i, label in  enumerate(["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$","Total loss $\mathcal{L}$" ]):
		plt.plot(train_history[:,i], label=label)
	plt.legend()
	plt.show()

	### Projecting into latent space
	Y0_ = model.tau(X_[:,0]).numpy()

	os.makedirs('data/generated/saved_Y/fish', exist_ok=True)
	np.savetxt('data/generated/saved_Y/fish/Y0__' + algorithm + '_fish_' + str('F2'), Y0_)
	np.savetxt('data/generated/saved_Y/fish/B__' + algorithm + '_fish_' + str('F2'), B_)

	Y0_ = np.loadtxt('data/generated/saved_Y/fish/Y0__BunDLeNet_fish_F2')
	B_ = np.loadtxt('data/generated/saved_Y/fish/B__BunDLeNet_fish_F2')

	fig = plt.figure(figsize=(4, 4))
	ax = plt.axes(projection='3d')
	true_y_line = ax.scatter(Y0_[:, 0], Y0_[:, 1], Y0_[:, 2], c=B_[:,0])
	plt.colorbar(true_y_line)
	plt.show()
