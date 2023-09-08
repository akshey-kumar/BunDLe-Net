import sys
sys.path.append(r'../')
import numpy as np
from functions import *
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
os.chdir('..')

"""

This script was used for plotting all the figures
in the jounral paper for BunDLe-Net. Some of the 
data was already produced in other python scripts 
can be found in the main repo unless otherwise 
indicated. To reproduce a figure, uncomment the 
section of code for the corresponding figure.

"""

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
model.build(input_shape=X_.shape)
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=0.001)
## Uncomment only if you want to train your model from scratch
# loss_array = train_model(X_,
#            B_,
#            model,
#            optimizer,
#            gamma=0.9, 
#            n_epochs=1000,
#            pca_init=True)



### figure 1 - bundle net on many worms (see scripts/embedding_worms_separately.py)
algorithm = 'BunDLeNet'
elev = [-45,0,22,-40,-35]
azim = [162,8,-105,65,101]

for worm_num in range(5):
    print(worm_num)
    Y0_ = np.loadtxt('data/generated/saved_Y/new_runs_3000_epochs/Y0__' + algorithm + '_worm_' + str(worm_num))
    B_ = np.loadtxt('data/generated/saved_Y/new_runs_3000_epochs/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)
    #plot_phase_space(Y0_, B_, state_names, show_points=False, legend=False)
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elev[worm_num], azim=azim[worm_num], roll=0)
    plot_ps_(fig, ax, Y=Y0_, B=B_, state_names=state_names, show_points=False, legend=True)
    #plt.savefig('figures/figure_1/Y0_' + algorithm + '_worm_' + str(worm_num) + '.pdf', transparent=True)
    plt.show()
    rotating_plot(Y=Y0_, B=B_, filename='figures/rotation_'+ algorithm + '_worm_'+str(worm_num) +'.gif', state_names=state_names, legend=True, show_points=False)

'''
### figure 1 - attempt 2 (see scripts/comparable_embeddings.py)
algorithm = 'BunDLeNet'
elev = -161
azim = 61

for worm_num in range(5):
    print(worm_num)
    Y0_ = np.loadtxt('data/generated/saved_Y/comparable_embeddings/Y0__' + algorithm + '_worm_' + str(worm_num))
    B_ = np.loadtxt('data/generated/saved_Y/comparable_embeddings/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elev, azim=azim, roll=0)
    plot_ps_(fig, ax, Y=Y0_, B=B_, state_names=state_names, show_points=False, legend=False)
    #plt.savefig('figures/figure_1/comparable_embeddings/Y0_' + algorithm + '_worm_' + str(worm_num) + '.pdf', transparent=True)
    #rotating_plot(Y=Y0_, B=B_, filename='figures/comparable_embeddings/rotation_'+ algorithm + '_worm_'+str(worm_num) +'.gif', state_names=state_names, legend=False, show_points=False)
plt.show()

### figure 2 - comparison of various algorithms
elev = [-117, 171, 94, 38, 27, 27, -22, -148]
azim = [-66, -146, -142, -146, -128, -119, -41, 161]

worm_num = 0
algorithms = ['PCA', 'tsne', 'autoencoder', 'autoregressor', 'cebra_B', 'cebra_time', 'cebra_hybrid', 'AbCNet']
for i, algorithm in enumerate(algorithms):
    print(i, algorithm)
    Y0_ = np.loadtxt('data/generated/saved_Y/comparison_algorithms/Y0_tr__' + algorithm + '.csv')
    B_ = np.loadtxt('data/generated/saved_Y/comparison_algorithms/B_train_1__' + algorithm + '.csv').astype(int)
    Y0_ = 2*Y0_/np.std(Y0_)
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elev[i], azim=azim[i], roll=0)
    plot_ps_(fig, ax, Y=Y0_, B=B_, state_names=state_names, show_points=False, legend=False)
    plt.savefig('figures/figure_2/Y0_' + algorithm + '.pdf', transparent=True)
    plt.show()
    #rotating_plot(Y=Y0_, B=B_, filename='figures/rotation_'+ algorithm + '_worm_'+str(worm_num) +'_pts.gif', state_names=state_names, legend=False, show_points=False)



### figure 3 - see dcc-methods/Results for paper/plotting


'''
### figure 4 - confusion matrix of behaviour and plot of true vs predicted dynamics
## Behaviour predictor (implicit in the BunDLe Net)
model.load_weights('data/generated/BunDLeNet_model')
Y0_ = model.tau(X_[:,0]).numpy() # Y_t
Y1_ = model.tau(X_[:,1]).numpy()
B_pred = model.predictor(Y1_).numpy().argmax(axis=1)
cf_matrix = confusion_matrix(B_, B_pred)

plt.figure(figsize=(5, 5))
heatmap = sns.heatmap(cf_matrix, cmap='Blues', linewidths=1, annot=True, fmt='g', xticklabels=state_names, yticklabels=state_names, square=True)
heatmap.set_ylabel('True Label')
heatmap.set_xlabel('Predicted Label')
plt.tick_params(axis='both', which='both', length=0)
plt.setp(heatmap.get_xticklabels(), rotation=45, ha='right')
plt.subplots_adjust(bottom=0.25, left=0.25)  # You can modify this value as needed
plt.savefig('figures/confusion_matrix.pdf', transparent=True)
plt.show()

## Dynamics model (implicit in the BunDLe Net)
Y1_pred = Y0_ + model.T_Y(Y0_).numpy()
fig = plt.figure(figsize=(8, 8))
ax = plt.axes(projection='3d')
ax.view_init(elev=37, azim=177, roll=0)
true_y_line = ax.plot(Y1_[:, 0], Y1_[:, 1], Y1_[:, 2], color='gray', linewidth=.8, linestyle='--', label=r'True $Y_{t+1}$') 
predicted_y_line = ax.plot(Y1_pred[:, 0], Y1_pred[:, 1], Y1_pred[:, 2], color='k',  linewidth=.8,  label=r'Predicted $Y_{t+1}$')
ax.set_axis_off()  
plt.legend(handles=[true_y_line[0], predicted_y_line[0]])

plt.savefig('figures/dynamics.pdf', transparent=True)
plt.show()
'''

### figure 5 - Learning curve
# Training losses vs epochs
loss_array = train_model(X_,
           B_,
           model,
           optimizer,
           gamma=0.9, 
           n_epochs=1000,
           pca_init=True)

plt.figure(figsize=(5,3.5))
for i, label in  enumerate(["$\mathcal{L}_{{Markov}}$", "$\mathcal{L}_{{Behavior}}$","Total loss $\mathcal{L}$" ]):
    plt.semilogy(loss_array[:,i], label=label)
plt.legend()
plt.xlabel('Training epochs')
plt.subplots_adjust(bottom=0.25)
plt.grid(axis='y', linestyle=':', color='gray', alpha=0.5)
plt.savefig('figures/learning_curve.pdf', transparent=True)
plt.show()



### figure 6 - Learning process - see BunDLe-Net/scripts/learning_process.py
epochs = 1000
delta_epochs = 50
for i in range(delta_epochs,epochs,delta_epochs):
    print(i)
    Y0_ = np.load('data/generated/learning_process/Y0_after_' + str(i) + '_epochs.npy')
    fig = plt.figure(figsize=(8, 8))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=56, azim=-2, roll=0)
    plot_ps_(fig, ax, Y=Y0_, B=B_, state_names=state_names, show_points=False, legend=False)
    plt.savefig('figures/learning_process/Y0_after_' + str(i) + '_epochs.pdf')


### figure 7 - distinct behavioural motifs
algorithm = 'BunDLeNet'
elev = [13, -47]
azim = [-148, 60]
for i, worm_num in enumerate([0, 0]):
    print(worm_num)
    Y0_ = np.loadtxt('data/generated/saved_Y/comparable_embeddings/Y0__' + algorithm + '_worm_' + str(worm_num))
    B_ = np.loadtxt('data/generated/saved_Y/comparable_embeddings/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elev[i], azim=azim[i], roll=0)
    plot_ps_(fig, ax, Y=Y0_, B=B_, state_names=state_names, show_points=False, legend=True)
    plt.savefig('figures/figure_7_distinct_motifs/Y0_' + algorithm + '_worm_' + str(worm_num) + 'per' + str(i) +'.pdf', transparent=True)
    #rotating_plot(Y=Y0_, B=B_, filename='figures/comparable_embeddings/rotation_'+ algorithm + '_worm_'+str(worm_num) +'.gif', state_names=state_names, legend=False, show_points=False)
'''