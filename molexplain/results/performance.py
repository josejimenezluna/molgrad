import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kde
from sklearn.metrics import roc_auc_score, roc_curve

from molexplain.train import N_FOLDS, rmse
from molexplain.utils import DATA_PATH, FIG_PATH

matplotlib.use("Agg")

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"],
        "font.size": 24,
    }
)

LABEL_GUIDE = {
    "ppb": r"$F_b(\%)$",
    "caco2": r"$-\log_{10} P_\mathrm{app}$ (cm/s)",
    "herg": r"$p\mathrm{IC}_{50}$ (hERG)",
}

sigmoid = lambda x: 1 / (1 + np.exp(-x))

if __name__ == "__main__":
    os.makedirs(FIG_PATH, exist_ok=True)
    # Regression plot for ppb, caco2, herg
    f, axs = plt.subplots(1, len(LABEL_GUIDE) + 1, figsize=(18, 6))

    for idx_plot, data in enumerate(LABEL_GUIDE.keys()):
        y_test = []
        y_hat = []

        rs = []
        rmses = []

        for idx_split in range(N_FOLDS):
            y = np.load(
                os.path.join(DATA_PATH, f"{data}", f"{data}_noHs_y_fold{idx_split}.npy")
            ).flatten()
            yh = np.load(
                os.path.join(
                    DATA_PATH, f"{data}", f"{data}_noHs_yhat_fold{idx_split}.npy"
                )
            ).flatten()

            rs.append(np.corrcoef(y, yh)[0, 1])
            rmses.append(rmse(y, yh))

            y_test.extend(y.tolist())
            y_hat.extend(yh.tolist())

        print(
            "Dataset {} | R: {:.3f} ± {:.3f} | RMSE: {:.3f} ± {:.3f}".format(
                data, np.mean(rs), np.std(rs), np.mean(rmses), np.std(rmses)
            )
        )

        y_test = np.array(y_test)
        y_hat = np.array(y_hat)

        # axs[idx_plot].scatter(y_test, y_hat, s=1.7)
        nbins = 300
        k = kde.gaussian_kde([y_test, y_hat])
        xi, yi = np.mgrid[
            y_test.min() : y_test.max() : nbins * 1j,
            y_hat.min() : y_hat.max() : nbins * 1j,
        ]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))

        im = axs[idx_plot].pcolormesh(xi, yi, zi.reshape(xi.shape), cmap="cividis")
        # e = f.colorbar(matplotlib.cm.ScalarMappable(), ax=axs[idx_plot])
        fmt = matplotlib.ticker.ScalarFormatter()
        fmt.set_scientific(True)
        fmt.set_powerlimits((-2, 4))
        plt.colorbar(im, ax=axs[idx_plot], format=fmt)

        axs[idx_plot].set_ylabel(f"Pred. {LABEL_GUIDE[data]}")
        axs[idx_plot].set_xlabel(f"Exp. {LABEL_GUIDE[data]}")
        # axs[idx_plot].grid(alpha=0.5)

    # cyp values

    y_test = []
    y_hat = []

    aucs = []

    for idx_split in range(N_FOLDS):
        y = np.load(
            os.path.join(DATA_PATH, "cyp", f"cyp_noHs_y_fold{idx_split}.npy")
        ).flatten()
        yh = np.load(
            os.path.join(DATA_PATH, "cyp", f"cyp_noHs_yhat_fold{idx_split}.npy")
        ).flatten()
        y_test.append(y)
        y_hat.append(yh)
        aucs.append(roc_auc_score(y, yh))

    print("Dataset cyp | AUC: {:.3f} ± {:.3f}".format(np.mean(aucs), np.std(aucs)))

    y_test = np.concatenate(y_test)
    y_hat = np.concatenate(y_hat)

    fpr, tpr, _ = roc_curve(y_test, y_hat)
    axs[3].plot(fpr, tpr, lw=1)
    axs[3].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axs[3].grid(alpha=0.5)
    axs[3].set_xlabel("Specificity (CYP3A4 inhibition)")
    axs[3].set_ylabel("Sensitivity (CYP3A4 inhibition)")

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_PATH, "graph_performance.pdf"))
    plt.close()
