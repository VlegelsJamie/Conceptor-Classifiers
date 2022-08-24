"""
This module implements various functions for creating the various data and model performance
visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt

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

    height = [67.17, 66.82, 65.93, 65.63, 64.69, 64.21, 63.62, 63.15, 60.28, 59.05, 58.15, 57.82, 56.96, 56.17, 54.76,
              53.84, 53.12, 50.58, 49.57, 46.33, 43.62, 42.94, 34.31, 33.97]
    bars = ('$ESN_{\mathsf{One-hot}}$', '$C_{\mathsf{Reduced}}$', '$C_{\mathsf{Combined}}$',
            '$C_{\mathsf{Forecast}}$', '$C_{\mathsf{Tubes}}$', '$C_{\mathsf{Reservoir}}$', 'MUSE', 'ROCKET', 'mrseql',
            '$ESN_{\mathsf{Forecast}}$', 'gRSF', 'STC', 'DTW_A', 'CIF', 'DTW_D', 'HC', '$C_{\mathsf{Standard}}$',
            'RISE', 'DTW_I', 'TapNet', 'CBOSS', 'ResNet', 'TSF', 'IT')

    fig, ax = plt.subplots()

    colors = plt.cm.Blues(np.linspace(0, 1, 9))

    # Save the chart so we can loop through the bars below.
    bars = ax.barh(np.flip(bars), np.flip(height), color=[
        colors[3], colors[3], colors[3], colors[3], colors[3], colors[3], colors[3], colors[7], colors[3], colors[3],
        colors[3], colors[3], colors[3], colors[3], colors[7], colors[3], colors[3], colors[3], colors[7], colors[7],
        colors[7], colors[7], colors[7], colors[7]]
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


if __name__ == '__main__':
    X_train, y_train = get_train_data()
    X_test, y_test = get_test_data()
    X_train_processed, X_test_processed = preprocess(X_train.copy(), X_test.copy())

    plot_example_samples(X_train)
    plot_preprocessed_samples(X_train, X_train_processed)
    plot_distribution_pie(y_train)
    plot_sota_graph()
