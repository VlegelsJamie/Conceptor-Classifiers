"""
This module implements various functions for creating the various data and model performance
visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from skopt.plots import _evaluate_min_params, _map_categories, partial_dependence_1D
from skopt.space import Categorical

from utils.data import get_train_data, get_test_data, preprocess


def plot_example_samples(X):
    """ Plots exemplary light curves for each class.

    :param X: Samples to plot a subset of.
    """

    time = np.arange(0, 36)

    labels = ['Mira Variable', 'KN', 'µ-lensing', 'SLSN', 'SNIax', 'SNIa-91bg', 'RR Lyrae', 'AGN', 'TDE', 'SNIbc',
              'EBE', 'M-dwarf', 'SNII', 'SNIa']

    idxs = [874, 1033, 21, 2437, 816, 1394, 2353, 1453, 37, 993, 179, 1076, 432, 1567]

    colors = plt.cm.Blues(np.linspace(0, 1, 9))

    fig, axes = plt.subplots(5, 3)
    for idx, ax in enumerate(fig.get_axes()):
        if idx != 14:
            ax.plot(time, X[idxs[idx], :, 0], label='u band', color=colors[3])
            ax.plot(time, X[idxs[idx], :, 1], label='g band', color=colors[4])
            ax.plot(time, X[idxs[idx], :, 2], label='r band', color=colors[5])
            ax.plot(time, X[idxs[idx], :, 3], label='i band', color=colors[6])
            ax.plot(time, X[idxs[idx], :, 4], label='z band', color=colors[7])
            ax.plot(time, X[idxs[idx], :, 5], label='y band', color=colors[8])

    for ax, title in zip(fig.get_axes(), labels):
        ax.set_title(title)

    fig.set_size_inches(10.5, 12.5)
    fig.delaxes(axes[4][2])
    fig.tight_layout()
    plt.savefig('figures/example_light_curves.pdf')
    plt.close()


def plot_preprocessed_samples(X, X_processed):
    """ Display figure of unprocessed compared to preprocessed samples.

    :param X: unprocessed training samples to plot a subset of.
    :param X_processed: processed training samples to plot a subset of.
    """

    plot_subset = np.array([X[100], X[2336], X[1453], X_processed[100], X_processed[2336], X_processed[1453]])

    fig, axes = plt.subplots(nrows=2, ncols=3)
    time = np.arange(0, 36)
    columns = ['TDE', 'RR Lyrae', 'AGN']

    colors = plt.cm.Blues(np.linspace(0, 1, 9))

    for i, ax in enumerate(fig.get_axes()):
        ax.plot(time, plot_subset[i, :, 0], label='u band', color=colors[3])
        ax.plot(time, plot_subset[i, :, 1], label='g band', color=colors[4])
        ax.plot(time, plot_subset[i, :, 2], label='r band', color=colors[5])
        ax.plot(time, plot_subset[i, :, 3], label='i band', color=colors[6])
        ax.plot(time, plot_subset[i, :, 4], label='z band', color=colors[7])
        ax.plot(time, plot_subset[i, :, 5], label='y band', color=colors[8])

        if i == 0:
            ax.set_ylim([-50, 300])
        if i == 1:
            ax.set_ylim([-1000, 2000])
        if i == 2:
            ax.set_ylim([-250, 200])
        if i > 2:
            ax.set_ylim([-1, 1])

    for ax, column in zip(axes[0], columns):
        ax.set_title(column)

    fig.set_size_inches(10, 6)
    fig.tight_layout()
    plt.savefig('figures/preprocessing_graph.pdf')
    plt.close()


def plot_distribution_pie(y):
    """ Creates a pie chart of class distribution of test data.

    :param y: test class labels to plot the class distribution of.
    """

    nr_class_samples_test = np.bincount(y)

    class_labels = ['µ-lensing', 'TDE', 'EBE', 'SNII', 'SNIax', 'Mira Variable', 'SNIbc', 'KN', 'M-dwarf', 'SNIa-91bg',
                    'AGN', 'SNIa', 'RR Lyrae', 'SLSN']

    plt.pie(x=nr_class_samples_test, labels=class_labels)
    plt.savefig('figures/class_distribution_pie.pdf')
    plt.close()


def plot_sota_graph():
    """
    Graph which compares conceptor classifier performance with state-of-the-art classifier
    performance.
    """

    height = [64.53, 64.21, 63.62, 63.47, 63.15, 61.63, 60.28, 58.15, 57.82, 56.96, 56.17, 54.76, 54.74, 53.84, 53.76,
              50.58, 49.57, 46.33, 43.62, 42.94, 34.31, 33.97]
    bars = ('$C_{\mathsf{timestep}}$', '$C_{\mathsf{reservoir}}$', 'MUSE', '$C_{\mathsf{classLabel}}$', 'ROCKET',
            '$C_{\mathsf{signalPred}}$', 'mrseql', 'gRSF', 'STC', 'DTW_A', 'CIF', 'DTW_D', '$C_{\mathsf{concat}}$',
            'HC', '$C_{\mathsf{combined}}$', 'RISE', 'DTW_I', 'TapNet', 'CBOSS', 'ResNet', 'TSF', 'IT')

    fig, ax = plt.subplots()

    colors = plt.cm.Blues(np.linspace(0, 1, 9))

    # Save the chart so we can loop through the bars below.
    bars = ax.barh(np.flip(bars), np.flip(height), color=[
        colors[3], colors[3], colors[3], colors[3], colors[3], colors[3], colors[3], colors[7],
        colors[3], colors[7], colors[3], colors[3], colors[3], colors[3], colors[3], colors[3],
        colors[7], colors[3], colors[7], colors[3], colors[7], colors[7]]
                   )

    plt.margins(x=0.08)

    # Axis formatting.
    ax.set_axisbelow(True)
    ax.xaxis.grid(True)

    # Add text annotations to the top of the bars.
    for bar in bars:
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + 0.15,
            round(bar.get_width(), 1),
            color="black",
            fontsize="small",
            weight="semibold",
        )

    # Add labels and a title. Note the use of `labelpad` and `pad` to add some
    # extra space between the text and the tick labels.
    ax.set_xlabel('Accuracy (%)')
    ax.set_ylabel('Model')

    fig.tight_layout()

    plt.savefig('figures/performance_sota_graph.pdf')
    plt.close()


def plot_partial_dependence(result, n_points=40, n_samples=250, dimensions=None, sample_source='random',
                            minimum='result', n_minimum_search=None, plot_dims=None):
    """
    Partial dependence graph adapted from the sklearn function "plot_objective".
    Only the diagonal of the original plot is used to show effect of a single dimension on
    the objective function

    :param result: Objective function result.
    :param n_points: Number of points at which to evaluate the partial dependence along
                     each dimension.
    :param n_samples: Number of samples to use for averaging the model function at each of
                      the n_points when sample_method is set to ‘random’.
    :param dimensions: Labels of the dimension variables.
    :param sample_source: Defines to samples generation to use for averaging the model
                          function at each of the n_points.
    :param minimum: Defines the values for the red points in the plots.
    :param n_minimum_search: Determines how many points should be evaluated to find the
                             minimum when using ‘expected_minimum’ or ‘expected_minimum_random’
    :param plot_dims: List of dimension names or dimension indices from the search-space
                      dimensions to be included in the plot.
    """

    space = result.space
    # Get the relevant search-space dimensions.
    if plot_dims is None:
        # Get all dimensions.
        plot_dims = []
        for row in range(space.n_dims):
            if space.dimensions[row].is_constant:
                continue
            plot_dims.append((row, space.dimensions[row]))
    else:
        plot_dims = space[plot_dims]
    # Number of search-space dimensions we are using.
    n_dims = len(plot_dims)
    if dimensions is not None:
        assert len(dimensions) == n_dims
    else:
        dim_labels = ["$X_{%i}$" % i if d.name is None else d.name
                      for i, d in plot_dims]
    x_vals = _evaluate_min_params(result, minimum, n_minimum_search)
    if sample_source == "random":
        x_eval = None
        samples = space.transform(space.rvs(n_samples=n_samples))
    else:
        x_eval = _evaluate_min_params(result, sample_source,
                                      n_minimum_search)
        samples = space.transform([x_eval])
    x_samples, minimum, _ = _map_categories(space, result.x_iters, x_vals)

    fig, ax = plt.subplots(7, 2, figsize=(2.7 * 2, 2.3 * 7))
    fig.tight_layout()
    fig.subplots_adjust(top=0.96, hspace=0.3, wspace=0.4, right=0.9, left=0.05)

    iscat = [isinstance(dim[1], Categorical) for dim in plot_dims]

    for i in range(7):
        for j in range(2):
            index, dim = plot_dims[i + (7 * j)]
            xi, yi = partial_dependence_1D(space, result.models[-1],
                                           index,
                                           samples=samples,
                                           n_points=n_points)
            ax[i, j].plot(xi, yi)
            ax[i, j].axvline(minimum[index], linestyle="--", color="r", lw=1)

    # Get ylim for all diagonal plots.
    ylim = [ax[i, j].get_ylim() for i in range(7) for j in range(2)]

    # Separate into two lists with low and high ylim.
    ylim_lo, ylim_hi = zip(*ylim)
    ylim_min = np.min(ylim_lo)
    ylim_max = np.max(ylim_hi)

    for i in range(7):
        for j in range(2):
            ax[i, j].set_ylim(*(ylim_min, ylim_max))
            index_i, dim_i = plot_dims[i]
            low, high = dim_i.bounds
            ax[i, j].set_xlim(low, high)
            ax[i, j].yaxis.tick_right()
            ax[i, j].yaxis.set_ticks_position('both')

            ax[i, j].xaxis.tick_top()
            ax[i, j].xaxis.set_label_position('top')
            ax[i, j].set_xlabel(dim_labels[i + (7 * j)])

            if dim_i.prior == 'log-uniform':
                ax[i, j].set_xscale('log')
            else:
                ax[i, j].xaxis.set_major_locator(MaxNLocator(6, prune='both', integer=iscat[i]))

    plt.show()
    plt.close()


if __name__ == '__main__':
    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()
    X_train_processed, X_test_processed = preprocess(X_train.copy(), X_test.copy())

    plot_example_samples(X_train)
    plot_preprocessed_samples(X_train, X_train_processed)
    plot_distribution_pie(y_train)
    plot_sota_graph()
