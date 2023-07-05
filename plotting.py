import sys
sys.path.append(r'../')
import numpy as np
from functions import *

algorithm = 'BunDLeNet'
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']
'''
### figure 1 - bundle net on many worms
elev = [-45,0,22,-40,-35]
azim = [162,8,-105,65,101]

for worm_num in range(5):
    print(worm_num)
    Y0_ = np.loadtxt('data/generated/saved_Y/new_runs/Y0__' + algorithm + '_worm_' + str(worm_num))
    B_ = np.loadtxt('data/generated/saved_Y/new_runs/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)
    #plot_phase_space(Y0_, B_, state_names, show_points=False, legend=False)
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.view_init(elev=elev[worm_num], azim=azim[worm_num], roll=0)
    plot_ps_(fig, ax, Y=Y0_, B=B_, state_names=state_names, show_points=False, legend=True)
    #plt.savefig('figures/figure_1/Y0_' + algorithm + '_worm_' + str(worm_num) + '.pdf', transparent=True)
    #rotating_plot(Y, B, state_names, show_points=True, legend=True, filename='rotation.gif', **kwargs)
    plt.show()
'''

### figure 2 - comparison of various algorithms
elev = [64, 171, 94, 38, 27, 27, -22, -148]
azim = [-126, -146, -142, -146, -128, -119, -41, 161]

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
    #rotating_plot(Y, B, state_names, show_points=True, legend=True, filename='rotation.gif', **kwargs)
    plt.show()