import numpy as np
from scipy import stats, ndimage
from _fnndeconv import deconvolve


def make_fake_data(ncells, nframes, dt=(1. / 50), rate=0.5, tau=1.,
                   sigma=0.2):
    """
    Generate fake fluorescence traces

    Arguments:
    ---------------------------------------------------------------------------
        ncells:     number of traces to generate
        nframes:    number of timebins to simulate
        dt:         timestep (s)
        rate:       spike rate (Hz)
        tau:        time constant of decay in calcium concentration (s)
        sigma:      SD of additive noise on fluorescence

    Returns:
    ---------------------------------------------------------------------------
        S:          spike counts
        C:          calcium concentration
        F:          simulated fluorescence

    Each of the outputs are (ncells, nframes) arrays
    """

    # poisson spikes
    S = stats.poisson.rvs(rate * dt, size=(ncells, nframes))

    # internal calcium dynamics
    nk = int(10 * tau / dt)
    t = np.arange(nk) * dt
    kernel = np.exp(-t / tau)
    C = ndimage.convolve1d(S.astype(np.float64), kernel, axis=1,
                           origin=-nk / 2)

    # noise
    F = C + np.random.normal(loc=0., scale=sigma, size=C.shape)

    return (A.squeeze() for A in (S, C, F))

def make_fake_movie(nframes, mask_shape=(256, 256), mask_center=None,
                    bg_intensity=0.1, mask_sigma=30, dt=(1. / 50), rate=0.5,
                    tau=1., sigma=0.8):

    # poisson spikes
    S = stats.poisson.rvs(rate * dt, size=nframes)

    # internal calcium dynamics
    nk = int(10 * tau / dt)
    t = np.arange(nk) * dt
    kernel = np.exp(-t / tau)
    C = ndimage.convolve1d(S.astype(np.float64), kernel, axis=0,
                           origin=(-nk / 2))

    # pixel weights (sum to 1)
    nr, nc = mask_shape
    if mask_center is None:
        mask_center = (nc // 2., nr // 2.)
    a, b = mask_center
    y, x = np.ogrid[:nr, :nc]
    xs = (x - a) ** 2.
    ys = (y - b) ** 2.
    twoss = 2. * mask_sigma ** 2.
    mask = (1. / (twoss * np.pi)) * np.exp(-1 * ((xs / twoss) + (ys / twoss)))

    # background fluorescence
    background_fluor = np.random.randn(nr, nc) * bg_intensity

    gamma = np.exp(-dt / tau)
    alpha = mask.ravel()
    beta = background_fluor.ravel()
    lamb = rate

    # spatially & temporally white noise
    epsilon = np.random.normal(loc=0., scale=sigma, size=C.shape)

    # simulated fluorescence
    F = C[None, :] * alpha[:, None] + beta[:, None] + epsilon

    theta = (sigma, alpha, beta, lamb, gamma)

    return F, C, S, theta


def make_demo_plots():

    np.random.seed(0)

    s, c, f = make_fake_data(1, 10000)
    n_best, c_best, ll_best, theta_best = deconvolve(f, verbosity=1)

    try:
        import matplotlib
        from matplotlib import pyplot as plt
        plt.ion()
    except ImportError:
        print "matplotlib is required for making the demo plots"
        return

    fig, ax = plt.subplots(2, 1, sharex=True)
    for aa in ax:
        aa.hold(True)

    t = np.arange(10000) * 0.02

    real, = ax[0].plot(t, f, 'b')
    inferred, = ax[0].plot(t, c_best + theta_best[1], 'r')
    ax[0].set_ylabel('Fluorescence')

    ax[1].plot(t, s, 'b')
    ax[1].plot(t, n_best, 'r')

    ax[1].set_xlabel('Time (sec)')
    ax[1].set_ylabel('Spikes')

    plt.show()

if __name__ == "__main__":
    make_demo_plots()
