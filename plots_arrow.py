import sys
sys.path.append(r'../')
import numpy as np
from functions import *

algorithm = 'BunDLeNet'
worm_num = 0
Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']

#plot_trajectories_with_arrows(Y0_, B_)
plot_phase_space(Y0_, B_, state_names, show_points=False, legend=False)