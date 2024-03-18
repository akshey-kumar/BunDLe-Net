import numpy as np
import seaborn as sns
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib.animation as animation
from matplotlib.colors import ListedColormap

def plot_phase_space(Y, B, state_names, show_points=True, legend=True, **kwargs):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    plot_ps_discrete_(fig, ax, Y=Y, B=B, state_names=state_names, show_points=show_points, legend=legend, **kwargs)
    plt.show()
    return fig, ax

def plot_ps_discrete_(fig, ax, Y, B, state_names, show_points=True, legend=True, colors=None, **kwargs):
    if Y.shape[1] == 3:
        points = np.array(Y.T).T.reshape(-1, 1, 3)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        if colors is None:
            colors = sns.color_palette('deep', len(state_names))
        for segment, state in zip(segments, B[:-1]):
            p = ax.plot(segment.T[0], segment.T[1], segment.T[2], color=colors[state], **kwargs)
        ax.set_axis_off()  
    else:
        print("Error: Dimension of input array is not 3")
    if legend == True:
        # Create legend
        legend_elements = [Line2D([0], [0], color=c, lw=4, label=state) for c, state in zip(colors, state_names)]
        ax.legend(handles=legend_elements)
    elif legend == 'colorbar':
        # Create a colormap and colorbar
        cmap = cm.colors.ListedColormap(colors)
        norm = cm.colors.Normalize(vmin=0, vmax=len(state_names)-1)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ticks=range(len(state_names)))
        cbar.ax.set_yticklabels(state_names)
    if show_points == True:
        ax.scatter(Y[:,0], Y[:,1], Y[:,2], c=B, s=1, cmap = ListedColormap(colors))
    return fig, ax


def plot_ps_continuous_(fig, ax, Y, B, cmap='viridis'):
    
    def color_transform(a, cmap=cmap):
        normalized_value = (a - B.min()) / (B.max() - B.min())
        cmap = plt.get_cmap(cmap)
        rgba_color = cmap(normalized_value)
        rgb_color = rgba_color[:3]

        return rgb_color
    
    for i in range(len(Y) - 1):
        d = (Y[i+1] - Y[i])
        ax.quiver(Y[i, 0], Y[i, 1], Y[i, 2],
                  d[0], d[1], d[2],
                  color=color_transform(B[i]), arrow_length_ratio=0.1/np.linalg.norm(d), linewidths=1)
    ax.set_axis_off()  

    x = ax.scatter(Y[:, 0], Y[:, 1], Y[:, 2],
                   c=B,
                   cmap=cmap,
                   s=4)
    return fig, ax

def plot_phase_space_continuous(Y, B, cmap='viridis', **kwargs):
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.view_init(**kwargs)
    plot_ps_(fig, ax, Y, B, cmap)
    plt.show()
    return fig, ax