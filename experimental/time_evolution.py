import sys
sys.path.append(r'../')
import numpy as np
from functions import *

from sklearn.metrics import confusion_matrix
import seaborn as sns
import os

os.chdir('..')

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
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sus. reversal', 'Slowing', 'Ventral turn']

### Preprocess and prepare data for BundLe Net
time, X = preprocess_data(X, data.fps)
X_, B_ = prep_data(X, B, win=15)

### Deploy BunDLe Net
model = BunDLeNet(latent_dim=3)
# model.T_Y = tf.keras.Sequential([
#             layers.Dense(3, activation='relu'),
#             layers.Dense(5, activation='relu'),
#             layers.Dense(5, activation='relu'),
#             layers.Dense(3, activation='linear'),
#             layers.Normalization(axis=-1)
#             ])
model.build(input_shape=X_.shape)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)

# Uncomment only if you want to train your model from scratch
loss_array = train_model(X_,
			 B_,
			 model,
			 optimizer,
			 gamma=0.9, 
			 n_epochs=2000,
			 pca_init=True,
			 			 )
model.save_weights('data/generated/time_evolution_model')

### Dynamics model (implicit in the BunDLe Net)
model.load_weights('data/generated/time_evolution_model')
Y0_ = model.tau(X_[:,0]).numpy() # Y_t
Y1_ = model.tau(X_[:,1]).numpy()
Y1_pred = Y0_ + model.T_Y(Y0_).numpy()

fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.view_init(elev=37, azim=177, roll=0)
true_y_line = ax.plot(Y1_[:, 0], Y1_[:, 1], Y1_[:, 2], color='gray', linewidth=.8, linestyle='--', label=r'True $Y_{t+1}$') 
predicted_y_line = ax.plot(Y1_pred[:, 0], Y1_pred[:, 1], Y1_pred[:, 2], color='#377eb8',  linewidth=.8,  label=r'Predicted $Y_{t+1}$')
ax.set_axis_off()  
plt.legend(handles=[true_y_line[0], predicted_y_line[0]])

## Time evolution for some chosen points 
t_steps = 100
for i in np.arange(0,Y0_.shape[0], 1000):
	y_start = Y0_[i]
	y_t = y_start
	Y_evolved = np.zeros((100,3))
	for i in range(t_steps):
		y_t = y_t + model.T_Y(y_t.reshape(1,3)).numpy()
		Y_evolved[i] = y_t[0]
	ax.scatter(y_start[0], y_start[1], y_start[2], c='r')
	evolved_y_line = ax.scatter(Y_evolved[:, 0], Y_evolved[:, 1], Y_evolved[:, 2], color='k', s=1.5,  label=r'Simulated $Y_{t+1}$')

ax.set_axis_off()  
plt.legend(handles=[true_y_line[0], predicted_y_line[0]])
plt.show()

plot_phase_space(Y0_, B_, state_names)

