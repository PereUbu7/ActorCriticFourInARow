import matplotlib
import numpy as np
import pandas as pd
from collections import namedtuple
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

EpisodeStats = namedtuple("Stats",["episode_lengths", "episode_rewards", "episode_td_error", "episode_value_loss", "episode_policy_loss", "episode_kl_divergence"])


def make_visual(layer_weights):
    max_scale = layer_weights.max(axis=-1).max(axis=-1)[...,
                                                        np.newaxis, np.newaxis]
    min_scale = layer_weights.min(axis=-1).min(axis=-1)[...,
                                                        np.newaxis, np.newaxis]
    return (255 * (layer_weights - min_scale) /
            (max_scale - min_scale)).astype('uint8')

def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))



def plot_episode_stats(stats, filters, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot filters
    fig6 = plt.figure(figsize=(10,5))

    width, height, channels, nOfFilters = filters.shape

    mapWidth = 4
    mapHeight = int(nOfFilters/4)
    totalHeight = height*mapHeight+mapHeight
    totalWidth = width*mapWidth+mapWidth

    l = np.zeros((totalWidth, 0, 2))
    colorPadding = np.zeros((totalWidth,totalHeight,1))
    xPadding = np.ones((1,width,2))
    yPadding = np.ones((totalWidth,1,2))
    for y in range(mapHeight):
        lx1 = filters[(np.s_[:],) * 3 + (y*mapWidth,)]
        lx1 = np.concatenate((lx1, xPadding), axis=0)
        for x in range(mapWidth-1):
            lx1 = np.concatenate((lx1, filters[(np.s_[:],) * 3 + (x+1+y*mapWidth,)]), axis=0)
            lx1 = np.concatenate((lx1, xPadding), axis=0)
        l = np.concatenate((l, lx1), axis=1)
        l = np.concatenate((l, yPadding), axis=1)
    l = np.concatenate((l, colorPadding), axis=2)
    plt.imshow(l, interpolation='nearest')
    plt.title("Convolutional layer 1 filters")
    plt.show(fig6)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10,5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10,5))
    plt.plot(stats.episode_td_error)
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("TD error")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    # Plot time steps and episode number
    fig4 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_value_loss)
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Value loss")
    if noshow:
        plt.close(fig4)
    else:
        plt.show(fig4)

    # Plot time steps and episode number
    fig5 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_kl_divergence)
    plt.xlabel("Time Steps")
    plt.ylabel("Sum of KL Divergencies")
    plt.title("KL Divergence")
    if noshow:
        plt.close(fig5)
    else:
        plt.show(fig5)

    return fig1, fig2, fig3
