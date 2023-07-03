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

#X_train, X_test, B_train, B_test = timeseries_train_test_split(X_, B_)
loss_array = train_model(X_,
			 B_,
			 model,
			 optimizer,
			 gamma=0.9, 
			 n_epochs=1000,
			 pca_init=True
						 )

# Training losses vs epochs
plt.figure()

for i, label in  enumerate(["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$","Total loss $\mathcal{L}$" ]):
	plt.plot(loss_array[:,i], label=label)
plt.legend()
plt.show()

### Projecting into latent space
Y0_ = model.tau(X_[:,0]).numpy() 

algorithm = 'BunDLeNet'
# Save the weights
model.save_weights('data/generated/BunDLeNet_model')
# np.savetxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num), Y0_)
# np.savetxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num), B_)
# Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
# B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)

### Plotting latent space dynamics
plot_latent_timeseries(Y0_, B_, state_names)

plot_phase_space(Y0_, B_, state_names = state_names, colors = ['#1b9e77', '#d95f02', '#7570b3', '#e7298a', '#66a61e', '#e6ab02', '#a6761d', '#666666'] )
### Run to produce rotating 3-D plot
#rotating_plot(Y0_, B_,filename='rotation_'+ algorithm + '_worm_'+str(worm_num) +'.gif', state_names=state_names)

### Performing PCA on the latent dimension (to check if there are redundant or correlated components)
pca = PCA()
Y_pca = pca.fit_transform(Y0_)
plot_latent_timeseries(Y_pca, B_, state_names)

# Checking if the third PC shows any structure
#plot_latent_timeseries(Y_pca[:,2], B_, state_names)

### Behaviour predictor (implicit in the BunDLe Net)
Y0_ = model.tau(X_[:,0]).numpy() # Y_t
Y1_ = model.tau(X_[:,1]).numpy()
B_pred = model.predictor(Y1_).numpy().argmax(axis=1)
accuracy_score(B_pred, B_)
plt.show()

# Dynamics model (implicit in the BunDLe Net)
Y1_pred = Y0_ + model.T_Y(Y0_).numpy()
fig = plt.figure(figsize=(4, 4))
ax = plt.axes(projection='3d')
true_y_line = ax.plot(Y1_[:, 0], Y1_[:, 1], Y1_[:, 2], color='gray', linewidth=.6, linestyle='--', label=r'True $Y_{t+1}$') #label=r'$Y^U_{t+1} = \tau(X_{t+1})$')
predicted_y_line = ax.plot(Y1_pred[:, 0], Y1_pred[:, 1], Y1_pred[:, 2], color='#377eb8',  linewidth=.6,  label=r'Predicted $Y_{t+1}$')#label=r'$Y^L_{t+1} = T_Y(Y_t) $')
ax.set_axis_off()  
plt.legend(handles=[true_y_line[0], predicted_y_line[0]])
plt.show()

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
