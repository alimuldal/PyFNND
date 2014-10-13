import numpy as np
from matplotlib import pyplot as plt


def plot_fit(F, n_hat, C_hat, theta_hat, dt):

    sigma, alpha, beta, lamb, gamma = theta_hat

    fig = plt.figure()
    gs = plt.GridSpec(3, 1)
    ax1 = fig.add_subplot(gs[0:2])
    ax2 = fig.add_subplot(gs[2:], sharex=ax1)
    axes = np.array([ax1, ax2])

    t = np.arange(F.shape[1]) * dt
    F_hat = alpha[:, None] * C_hat[None, :] + beta[:, None]

    axes[0].hold(True)
    axes[0].plot(t, F.sum(0), '-b', label=r'$F$')
    axes[0].plot(t, F_hat.sum(0), '-r', lw=1,
                 label=r'$\hat{\alpha}\hat{C}+\hat{\beta}$')
    axes[0].legend(loc=1, fancybox=True, fontsize='large')
    axes[0].tick_params(labelbottom=False)

    axes[1].plot(t, n_hat, '-k')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel(r'$\hat{n}$', fontsize='large')
    axes[1].set_xlim(0, t[-1])

    fig.tight_layout()

    return fig, axes

def plot_fit_2D(F, n_hat, C_hat, theta_hat, dt, nrows, ncols):

    sigma, alpha, beta, lamb, gamma = theta_hat

    fig = plt.figure()
    gs = plt.GridSpec(6, 4)
    ax1 = fig.add_subplot(gs[0:4, 0:3])
    ax2 = fig.add_subplot(gs[4:6, 0:3], sharex=ax1)
    ax3 = fig.add_subplot(gs[0:3, 3])
    ax4 = fig.add_subplot(gs[3:6, 3], sharex=ax3, sharey=ax3)
    axes = np.array([ax1, ax2, ax3, ax4])

    t = np.arange(F.shape[1]) * dt
    F_hat = alpha[:, None] * C_hat[None, :] + beta[:, None]

    axes[0].hold(True)
    axes[0].plot(t, F.sum(0), '-b', label=r'$F$')
    axes[0].plot(t, F_hat.sum(0), '-r', lw=1,
                 label=r'$\hat{\alpha}\hat{C}+\hat{\beta}$')
    axes[0].legend(loc=1, fancybox=True, fontsize='large')
    axes[0].tick_params(labelbottom=False)

    axes[1].plot(t, n_hat, '-k')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel(r'$\hat{n}$', fontsize='large')
    axes[1].set_xlim(0, t[-1])

    for ax in axes[2:4]:
        ax.hold(True)
        ax.set_axis_off()

    bbox_props = dict(boxstyle="round", fc=[0, 0, 0, 0.5], ec="None")

    # inferred mask
    axes[2].imshow(alpha.reshape(nrows, ncols), interpolation='nearest',
               cmap=plt.cm.gray)
    axes[2].annotate(r'$\hat{\alpha}$', (0, 0), xycoords='data', xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top', color='y',
                 bbox=bbox_props, fontsize='x-large')

    # inferred background
    axes[3].imshow(beta.reshape(nrows, ncols), interpolation='nearest',
               cmap=plt.cm.gray)
    axes[3].annotate(r'$\hat{\beta}$', (0, 0), xycoords='data', xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top', color='y',
                 bbox=bbox_props, fontsize='x-large')

    fig.tight_layout()

    return fig, axes

def ground_truth_1D(F, n_hat, C_hat, theta_hat, n, C, theta, dt):

    sigma_hat, alpha_hat, beta_hat, lamb_hat, gamma_hat = theta_hat
    sigma, alpha, beta, lamb, gamma = theta

    alpha, beta = (np.atleast_1d(a) for a in (alpha, beta))

    fig = plt.figure()
    gs = plt.GridSpec(4, 1)
    ax1 = fig.add_subplot(gs[0:2])
    ax2 = fig.add_subplot(gs[2:3], sharex=ax1)
    ax3 = fig.add_subplot(gs[3:4], sharex=ax1)
    axes = np.array([ax1, ax2, ax3])

    t = np.arange(F.shape[1]) * dt
    F_real = alpha[:, None] * C[None, :] + beta[:, None]
    F_hat = alpha_hat[:, None] * C_hat[None, :] + beta_hat[:, None]

    axes[0].hold(True)

    # noisy fluorescence
    axes[0].plot(t, F.sum(0), '-k', alpha=0.5, label=r'$F$')

    # fitted fluorescence
    axes[0].plot(t, F_hat.sum(0), '-b', lw=1,
                 label=r'$\hat{\alpha}\hat{C}+\hat{\beta}$')

    # true noise-free fluorescence
    axes[0].plot(t, F_real.sum(0), '-r', lw=2, label=r'$\alpha C+\beta$')
    axes[0].legend(loc=1, ncol=3, fancybox=True, fontsize='large')
    axes[0].tick_params(labelbottom=False)

    # true spikes
    axes[1].plot(t, n, '-k', label=r'$n$')
    axes[1].set_ylabel(r'$n$', fontsize='large')
    axes[1].tick_params(labelbottom=False)

    # inferred spike probabilities
    axes[2].plot(t, n_hat, '-k', label=r'$\hat{n}$')

    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel(r'$\hat{n}$', fontsize='large')
    axes[2].set_xlim(0, t[-1])

    fig.tight_layout()

    return fig, axes


def ground_truth_2D(F, n_hat, C_hat, theta_hat, n, C, theta, dt, nrows, ncols):

    sigma_hat, alpha_hat, beta_hat, lamb_hat, gamma_hat = theta_hat
    sigma, alpha, beta, lamb, gamma = theta

    gs = plt.GridSpec(12, 4)

    fig = plt.figure()

    ax1 = fig.add_subplot(gs[0:6, 0:3])
    ax2 = fig.add_subplot(gs[6:9, 0:3], sharex=ax1)
    ax3 = fig.add_subplot(gs[9:12, 0:3], sharex=ax1)
    ax4 = fig.add_subplot(gs[0:3, 3])
    ax5 = fig.add_subplot(gs[3:6, 3], sharex=ax4, sharey=ax4)
    ax6 = fig.add_subplot(gs[6:9, 3], sharex=ax4, sharey=ax4)
    ax7 = fig.add_subplot(gs[9:12, 3], sharex=ax4, sharey=ax4)

    t = np.arange(F.shape[1]) * dt
    F_real = alpha[:, None] * C[None, :] + beta[:, None]
    F_hat = alpha_hat[:, None] * C_hat[None, :] + beta_hat[:, None]

    ax1.hold(True)

    # noisy fluorescence
    ax1.plot(t, F.sum(0), '-k', alpha=0.5, label=r'$F$')

    # fitted fluorescence
    ax1.plot(t, F_hat.sum(0), '-b', label=r'$\hat{\alpha}\hat{C}+\hat{\beta}$')

    # true noise-free fluorescence
    ax1.plot(t, F_real.sum(0), '-r',label=r'$\alpha C+\beta$')
    ax1.legend(loc=1, fancybox=True, ncol=3, fontsize='large')
    ax1.tick_params(labelbottom=False)

    # true spikes
    ax2.plot(t, n, '-k', label=r'$n$')
    ax2.set_ylabel(r'$n$', fontsize='large')
    ax2.tick_params(labelbottom=False)

    # inferred spike probabilities
    ax3.plot(t, n_hat, '-k', label=r'$\hat{n}$')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel(r'$\hat{n}$', fontsize='large')
    ax3.set_xlim(0, t[-1])

    for ax in (ax4, ax5, ax6, ax7):
        ax.hold(True)
        ax.set_axis_off()

    bbox_props = dict(boxstyle="round", fc=[0, 0, 0, 0.5], ec="None")

    # true mask
    ax4.imshow(alpha.reshape(nrows, ncols), interpolation='nearest',
               cmap=plt.cm.gray)
    ax4.annotate(r'$\alpha$', (0, 0), xycoords='data', xytext=(5, -5),
                 textcoords='offset points', ha='left',
                 va='top', color='y', bbox=bbox_props, fontsize='x-large')

    # inferred mask
    ax5.imshow(alpha_hat.reshape(nrows, ncols), interpolation='nearest',
               cmap=plt.cm.gray)
    ax5.annotate(r'$\hat{\alpha}$', (0, 0), xycoords='data', xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top', color='y',
                 bbox=bbox_props, fontsize='x-large')

    # true background
    ax6.imshow(beta.reshape(nrows, ncols), interpolation='nearest',
               cmap=plt.cm.gray)
    ax6.annotate(r'$\beta$', (0, 0), xycoords='data', xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top', color='y',
                 bbox=bbox_props, fontsize='x-large')

    # inferred background
    ax7.imshow(beta_hat.reshape(nrows, ncols), interpolation='nearest',
               cmap=plt.cm.gray)
    ax7.annotate(r'$\hat{\beta}$', (0, 0), xycoords='data', xytext=(5, -5),
                 textcoords='offset points', ha='left', va='top', color='y',
                 bbox=bbox_props, fontsize='x-large')

    ax_array = np.array((ax1, ax2, ax3, ax4, ax5, ax6, ax7))

    # fig.tight_layout()

    return fig, ax_array
