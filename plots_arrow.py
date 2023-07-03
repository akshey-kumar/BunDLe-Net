import sys
sys.path.append(r'../')
import numpy as np
from functions import *

'''
def plot_trajectories_with_arrows(X, B):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Set up colors for arrows based on B values
    cmap = plt.get_cmap('tab10')
    norm = plt.Normalize(vmin=0, vmax=np.max(B))
    colors = cmap(norm(B[:-1]))

    # Plot trajectories with colored arrows
    for i in range(len(X) - 1):
        d = (X[i+1] - X[i])
        if i%1==0:
            ax.quiver(X[i, 0], X[i, 1], X[i, 2],
                      d[0], d[1], d[2],
                      color=colors[i], arrow_length_ratio=0.06/np.linalg.norm(d), linewidths=1)
        else:
            ax.quiver(X[i, 0], X[i, 1], X[i, 2],
                      X[i+1, 0] - X[i, 0], X[i+1, 1] - X[i, 1], X[i+1, 2] - X[i, 2],
                      color=colors[i], arrow_length_ratio=0.0, linewidths=1)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Create a separate Axes for the colorbar
    cax = fig.add_axes([0.95, 0.1, 0.03, 0.8])
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cax)
    cbar.set_label('B')

    plt.show()


def plot_ps_(fig, ax, Y, B, state_names, show_points=True, legend=True, colors=None, **kwargs):
    if colors is None:
        colors = sns.color_palette('deep', len(state_names))

    for i in range(len(Y) - 1):
        d = (Y[i+1] - Y[i])
        ax.quiver(Y[i, 0], Y[i, 1], Y[i, 2],
                  d[0], d[1], d[2],
                  color=colors[B[i]], arrow_length_ratio=0.1/np.linalg.norm(d), linewidths=1, **kwargs)
    ax.set_axis_off()  

    if legend == True:
        # Create legend
        legend_elements = [Line2D([0], [0], color=c, lw=4, label=state) for c, state in zip(colors, state_names)]
        ax.legend(handles=legend_elements)

    if show_points == True:
        ax.scatter(Y[:,0], Y[:,1], Y[:,2], c=B, s=1, cmap = ListedColormap(colors))
    return fig, ax


def plot_phase_space(Y, B, state_names, show_points=True, legend=True, **kwargs):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    plot_ps_(fig, ax, Y=Y, B=B, state_names=state_names, show_points=show_points, legend=legend, **kwargs)
    plt.show()
    return fig, ax
'''
algorithm = 'BunDLeNet'
worm_num = 0
Y0_ = np.loadtxt('data/generated/saved_Y/Y0__' + algorithm + '_worm_' + str(worm_num))
B_ = np.loadtxt('data/generated/saved_Y/B__' + algorithm + '_worm_' + str(worm_num)).astype(int)
state_names = ['Dorsal turn', 'Forward', 'No state', 'Reverse-1', 'Reverse-2', 'Sustained reversal', 'Slowing', 'Ventral turn']

#plot_trajectories_with_arrows(Y0_, B_)
plot_phase_space(Y0_, B_, state_names, show_points=False, legend=False)