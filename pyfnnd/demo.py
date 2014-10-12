import numpy as np
from scipy import stats, signal
from _fnndeconv import deconvolve


def make_fake_data(ncells, nframes, dt=(1. / 50), rate=0.5, tau=1.,
                   sigma=0.2, alpha=1., beta=0., seed=None):
    """
    Generate 1D fake fluorescence traces

    Arguments:
    ---------------------------------------------------------------------------
        ncells:     number of traces to generate
        nframes:    number of timebins to simulate
        dt:         timestep (s)
        rate:       mean spike rate (Hz)
        tau:        time constant of decay in calcium concentration (s)
        sigma:      SD of additive noise on fluorescence
        alpha:      scaling parameter for fluorescence modulation
        beta:       baseline fluorescence

    Returns:
    ---------------------------------------------------------------------------
        F:          fluorescence [ncells, nframes]
        C:          calcium concentration [ncells, nframes]
        n:          spike train [ncells, nframes]
        theta:      tuple of true model parameters [5,]

    """

    gen = np.random.RandomState(seed)

    # poisson spikes
    n = gen.poisson(rate * dt, size=(ncells, nframes))

    # internal calcium dynamics
    gamma = np.exp(-dt / tau)
    C = signal.lfilter(np.r_[1], np.r_[1, -gamma], n, axis=1)

    # noise
    F = C + gen.randn(*C.shape) * sigma

    lamb = rate
    theta = (sigma, alpha, beta, lamb, gamma)

    return F, C, n, theta


def make_fake_movie(nframes, mask_shape=(256, 256), mask_center=None,
                    bg_intensity=0.1, mask_sigma=30, dt=(1. / 50), rate=0.5,
                    tau=1., sigma=0.8, seed=None):
    """
    Generate 2D fake fluorescence movie

    Arguments:
    ---------------------------------------------------------------------------
        nframes:        number of timebins to simulate
        mask_shape:     tuple (nrows, ncols), shape of a single movie frame
        mask_center:    tuple (x, y), pixel coords of cell center
        bg_intensity:   scalar, amplitude of (static) baseline fluorescence
        mask_sigma:     scalar, standard deviation of Gaussian mask
        dt:             timestep (s)
        rate:           mean spike rate (Hz)
        tau:            time constant of decay in calcium concentration (s)
        sigma:          SD of additive noise on fluorescence

    Returns:
    ---------------------------------------------------------------------------
        F:          fluorescence [npixels, nframes]
        C:          calcium concentration [nframes,]
        n:          spike train [nframes,]
        theta:      tuple of true model parameters [5,]

    """

    gen = np.random.RandomState(seed)

    # poisson spikes
    n = gen.poisson(rate * dt, size=nframes)

    # internal calcium dynamics
    gamma = np.exp(-dt / tau)
    C = signal.lfilter(np.r_[1], np.r_[1, -gamma], n, axis=0)

    # pixel weights (sum == 1)
    nr, nc = mask_shape
    npix = nr * nc
    if mask_center is None:
        mask_center = (nc // 2., nr // 2.)
    a, b = mask_center
    y, x = np.ogrid[:nr, :nc]
    xs = (x - a) ** 2.
    ys = (y - b) ** 2.
    twoss = 2. * mask_sigma ** 2.
    alpha = np.exp(-1 * ((xs / twoss) + (ys / twoss))).ravel()
    alpha /= alpha.sum()

    # mask = np.array([
    #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    #     [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    #     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    #     [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
    #     [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
    #     [0, 1, 0, 0, 1, 1, 0, 0, 1, 0],
    #     [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
    #     [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
    #     [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #     ]).astype(np.float64)
    # alpha = mask.ravel()
    # alpha /= alpha.sum()
    # npix = alpha.size

    # background fluorescence
    beta = gen.randn(npix) * bg_intensity

    # firing rate (spike probability per sec)
    lamb = rate

    # spatially & temporally white noise
    epsilon = gen.randn(npix, nframes) * sigma

    # simulated fluorescence
    F = C[None, :] * alpha[:, None] + beta[:, None] + epsilon

    theta = (sigma, alpha, beta, lamb, gamma)

    return F, C, n, theta


# def make_demo_plots():

#     np.random.seed(0)

#     s, c, f = make_fake_data(1, 10000)
#     n_best, c_best, ll_best, theta_best = deconvolve(f, verbosity=1)

#     try:
#         import matplotlib
#         from matplotlib import pyplot as plt
#         plt.ion()
#     except ImportError:
#         print "matplotlib is required for making the demo plots"
#         return

#     fig, ax = plt.subplots(2, 1, sharex=True)
#     for aa in ax:
#         aa.hold(True)

#     t = np.arange(10000) * 0.02

#     real, = ax[0].plot(t, f, 'b')
#     inferred, = ax[0].plot(t, c_best + theta_best[1], 'r')
#     ax[0].set_ylabel('Fluorescence')

#     ax[1].plot(t, s, 'b')
#     ax[1].plot(t, n_best, 'r')

#     ax[1].set_xlabel('Time (sec)')
#     ax[1].set_ylabel('Spikes')

#     plt.show()

if __name__ == "__main__":
    make_demo_plots()
