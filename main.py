import sys
sys.path.append(r'../')
import numpy as np
from functions import *

### Load Data (and excluding behavioural neurons)
worm_num = 2
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

loss_array = train_model(X_,
			 B_,
			 model,
			 optimizer,
			 gamma=0.9, 
			 n_epochs=2000,
			 pca_init=False,
			 best_of_5_init=True
						 )

# Training losses vs epochs
plt.figure()
for i, label in  enumerate(["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$","Total loss $\mathcal{L}$" ]):
	plt.semilogy(loss_array[:,i], label=label)
plt.legend()
plt.show()

### Projecting into latent space
Y0_ = model.tau(X_[:,0]).numpy() 

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
### Run to produce rotating 3-D plot
rotating_plot(Y0_, B_,filename='figures/rotation_'+ algorithm + '_worm_'+str(worm_num) +'.gif', state_names=state_names, legend=False)

### Performing PCA on the latent dimension (to check if there are redundant or correlated components)
pca = PCA()
Y_pca = pca.fit_transform(Y0_)
plot_latent_timeseries(Y_pca, B_, state_names)

### Mean pariwise distance analysis
# pd_Y = np.linalg.norm(Y0_[:, np.newaxis] - Y0_, axis=-1) < 0.8
# plt.matshow(pd_Y, cmap='Greys')
# #plt.colorbar()
# plot_latent_timeseries(Y0_, B_, state_names)
# plt.show()

# ### Linear response
# def linear_response(m):
#   linear_response = np.zeros_like(X_[0,0])
#   for i, y0_ in enumerate(Y0_):
#       linear_response += X_[i,0]*Y0_[i,m]
#   return linear_response
# plt.figure()
# plt.imshow(linear_response(0))
# plt.figure()
# plt.imshow(linear_response(1))
# plt.figure()
# plt.imshow(linear_response(2))
# plt.show()
