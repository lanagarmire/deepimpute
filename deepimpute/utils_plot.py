import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors.kde import KernelDensity


def train_test_scatter(data, n_max=10000, lims=None):
    fig, ax = plt.subplots(1, len(data.keys()))
    compt = 0
    maxVal = np.max(
        [
            max(np.percentile(x1, 99.5), np.percentile(x2, 99.5))
            for x1, x2 in data.values()
        ]
    )
    if lims is None:
        lims = [0, maxVal]
    for key, vect in data.items():
        y = np.reshape(np.array(vect[0]), -1)
        y_imp = np.reshape(np.array(vect[1]), -1)

        if len(y) > n_max:
            subIdx = np.random.choice(range(len(y)), n_max, replace=False)
            y = y[subIdx]
            y_imp = y_imp[subIdx]

        xy = np.vstack([y, y_imp]).T
        kde = KernelDensity(kernel="tophat", bandwidth=0.2).fit(xy)
        z = np.exp(kde.score_samples(xy))

        idx = z.argsort()
        # Plotting
        ax_i = ax
        if len(data.items()) > 1:
            ax_i = ax[compt]
        ax_i.scatter(y[idx], y_imp[idx], c=z[idx], edgecolor="", s=10)
        ax_i.set_xlabel("Raw")
        ax_i.set_ylabel("Imputated")
        ax_i.set_title(key)
        ax_i.plot(lims, lims, "r-.", linewidth=2)
        ax_i.set_xlim(lims)
        ax_i.set_ylim(lims)
        compt += 1
    plt.show()
    return ax
