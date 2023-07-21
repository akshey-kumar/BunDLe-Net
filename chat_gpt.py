import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
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
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sus. reversal', 'Slowing', 'Ventral turn']






# Define the rotation angles
elev = 20
azim = 30
def update_view(num, fig, axes, Y, B, state_names, show_points, legend):
    # Update the view angles for all the plots
    for ax in axes:
        ax.view_init(elev=elev, azim=azim + num)

algorithm = "BunDLeNet"

# Create subplots for each plot
num_plots = 5
fig, axes = plt.subplots(nrows=1, ncols=num_plots, figsize=(8*num_plots, 8), subplot_kw={'projection': '3d'})

# Loop through each plot
for worm_num, ax in zip(range(num_plots), axes):
    Y0_ = np.loadtxt('data/generated/saved_Y/comparable_embeddings/Y0__' + algorithm + '_worm_' + str(worm_num))
    B_ = np.loadtxt('data/generated/saved_Y/comparable_embeddings/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)

    plot_ps_(fig, ax, Y=Y0_, B=B_, state_names=state_names, show_points=False, legend=False)

# Create the animation
ani = FuncAnimation(fig, update_view, frames=range(360), fargs=(fig, axes, Y0_, B_, state_names, False, False), interval=50)
plt.show()