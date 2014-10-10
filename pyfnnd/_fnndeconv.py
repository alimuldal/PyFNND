import numpy as np
from scipy import ndimage, signal
from itertools import izip
import time
import warnings
from _tridiag_solvers import trisolve
from matplotlib import pyplot as plt

DTYPE = np.float64
EPS = np.finfo(DTYPE).eps


import ipdb
plt.ion()


# joblib is an optional dependency, required only for processing multiple cells
# in parallel
try:
    from joblib import Parallel, delayed

    def apply_all_cells(F, n_jobs=-1, disp=1, *fnn_args, **fnn_kwargs):
        """
        Run FNN deconvolution on multiple cells in parallel

        Arguments:
        -----------------------------------------------------------------------
        F: ndarray, [nc, nt]
            measured fluorescence values

        n_jobs: int scalar
            number of jobs to process in parallel. if n_jobs == -1, all cores
            are used.

        *fnn_args, **fnn_kwargs
            additional arguments to pass to deconvolve()

        Returns:
        -----------------------------------------------------------------------
        n_hat_best: ndarray, [nc, nt]
            MAP estimate of the most likely spike train

        C_hat_best: ndarray, [nc, nt]
            estimated intracellular calcium concentration (A.U.)

        LL: ndarray, [nc,]
            posterior log-likelihood of F given n_hat_best and theta_best

        theta_best: ndarray, [nc, 5]
            model parameters, updated according to learn_theta
        """

        F = np.atleast_2d(F)
        nc, nt = F.shape

        pool = Parallel(n_jobs=n_jobs, verbose=disp, pre_dispatch='n_jobs * 2')

        results = pool(delayed(deconvolve)
                       (rr, *fnn_args, **fnn_kwargs) for rr in F)
        n_hat, C_hat, LL, theta = (np.vstack(a) for a in izip(*results))

        return n_hat, C_hat, LL, theta

except ImportError:
    pass


def deconvolve(F, C0=None, theta0=None, dt=0.02, rate=0.5, tau=1.,
               learn_theta=(0, 0, 0, 0, 0), params_tol=1E-6, spikes_tol=1E-6,
               params_maxiter=20, spikes_maxiter=100, verbosity=0, plot=False):
    """
    Fast Non-Negative Deconvolution
    ---------------------------------------------------------------------------
    This function uses an interior point method to solve the following
    optimization problem:

        n_hat = argmax_{n >= 0} P(n | F)

    where n_hat_best is a maximum a posteriori estimate for the most likely
    spike train, given the fluorescence signal F, and the model:

    C_{t} = gamma * C_{t-1} + n_{t},            n_{t} ~ Poisson(lambda * dt)
    F_{t} = C_{t} + beta + epsilon,             epsilon ~ N(0, sigma)

    It is also possible to estimate the model parameters sigma, beta and lambda
    from the data using pseudo-EM updates.

    Arguments:
    ---------------------------------------------------------------------------
    F: ndarray, [nt] or [npix, nt]
        measured fluorescence values

    C0: ndarray, [nt]
        initial estimate of the calcium concentration for each time bin

    theta0: len(5) sequence
        initial estimates of the model parameters
        (sigma, alpha, beta, lambda, gamma).

    dt: float scalar
        duration of each time bin (s)

    rate: float scalar
        estimate of mean firing rate (Hz), ignored if theta0 is not None

    tau: float scalar
        estimate of calcium decay time constant (s), ignored if theta0 is not
        None

    learn_theta: len(5) bool sequence
        specifies which of the model parameters to attempt learn via pseudo-EM
        iterations. currently gamma cannot be optimised.

    spikes_tol: float scalar
        termination condition for interior point spike train estimation:
            params_tol > abs((LL_prev - LL) / LL)

    params_tol: float scalar
        as above, but for the model parameter estimation

    spikes_maxiter: int scalar
        maximum number of interior point iterations to estimate MAP spike train

    params_maxiter: int scalar
        maximum number of pseudo-EM iterations to estimate model parameters

    verbosity: int scalar
        0: no convergence messages (default)
        1: convergence messages for model parameters
        2: convergence messages for model parameters & MAP spike train

    Returns:
    ---------------------------------------------------------------------------
    n_hat_best: ndarray, [nt]
        MAP estimate of the most likely spike train

    C_hat_best: ndarray, [nt]
        estimated intracellular calcium concentration (A.U.)

    LL_best: float scalar
        posterior log-likelihood of F given n_hat_best and theta_best

    theta_best: len(5) tuple
        model parameters, updated according to learn_theta

    Reference:
    ---------------------------------------------------------------------------
    Vogelstein, J. T., Packer, A. M., Machado, T. A., Sippy, T., Babadi, B.,
    Yuste, R., & Paninski, L. (2010). Fast nonnegative deconvolution for spike
    train inference from population calcium imaging. Journal of
    Neurophysiology, 104(6), 3691-704. doi:10.1152/jn.01073.2009

    """

    tstart = time.time()

    F = np.atleast_2d(F)
    npix, nt = F.shape

    # ensure that C_hat is non-negative
    offset = F.min() - EPS
    F = F - offset

    if theta0 is None:
        theta = _init_theta(F, dt, rate=rate, tau=tau)
    else:
        sigma, alpha, beta, lamb, gamma = theta0
        # beta absorbs the offset
        beta = beta - offset
        theta = sigma, alpha, beta, lamb, gamma

    sigma, alpha, beta, lamb, gamma = theta

    if C0 is None:

        # it's necessary to vary the value of n0 according to the noise level.
        # when total noise is higher, n0 needs to be set higher or optimization
        # fails completely. however, when total noise is small then setting n0
        # too high will mean that n_best will have a significant baseline non-
        # zero spike probability (which probably ought to be absorbed by beta).
        n0 = 10 * np.ones(nt) * (sigma / np.sqrt(npix))
        C0 = signal.lfilter(np.r_[1], np.r_[1, -gamma], n0, axis=0)

    # if we're not learning the parameters, this step is all we need to do
    n_hat, C_hat, LL = _get_MAP_spikes(F, C0, theta, dt, spikes_tol,
                                       spikes_maxiter, verbosity)

    # pseudo-EM iterations to optimize the model parameters
    if np.any(learn_theta):

        if verbosity >= 1:
            print('Params: iter=%3i; LL=%12.2f; delta_LL= N/A' % (0, LL))

        nloop_params = 1
        done = False

        while not done:

            # update the parameter estimates
            theta1 = _update_theta(n_hat, C_hat, F, theta, dt, learn_theta)

            # get the new n_hat, C_hat, and LL
            n1, C_hat1, LL1 = _get_MAP_spikes(F, C_hat, theta1, dt, spikes_tol,
                                              spikes_maxiter, verbosity)

            # test for convergence
            delta_LL = -((LL1 - LL) / LL)

            if verbosity >= 1:
                print('params: iter=%3i; LL=%12.2f; delta_LL= %8.4g'
                      % (nloop_params, LL1, delta_LL))

            # if the LL gets significantly worse, keep the old parameters and
            # terminate now
            if delta_LL < - 1E-5:
                if verbosity >= 1:
                    print('Terminating parameter optimization after %i '
                          'iterations because LL is decreasing' %
                          nloop_params)
                done = True

            else:
                # keep the new parameters
                n_hat, C_hat, LL, theta = n1, C_hat1, LL1, theta1

                # check the other termination conditions
                if np.abs(delta_LL) < params_tol:
                    if verbosity >= 1:
                        print("Parameter optimization converged after %i "
                              "iterations" % nloop_params)
                    done = True

                elif nloop_params == params_maxiter:
                    if verbosity >= 1:
                        print('Parameter optimization failed to converge '
                              'before maxiter was reached (%i)' % nloop_params)
                    done = True

            if done:
                print "Last delta log-likelihood:\t%8.4g" % delta_LL
                print "Best posterior log-likelihood:\t%11.4f" % LL

            # increment the loop counter
            nloop_params += 1

    if verbosity >= 1:
        time_taken = time.time() - tstart
        print "Completed: %s" % _s2h(time_taken)

    sigma, alpha, beta, lamb, gamma = theta

    # correct for the offset we originally applied to F
    beta = beta + offset

    # since we can't use FNND to estimate the spike probabilities in the 0th
    # timebin, for convenience we just concatenate 0 to the start of
    # n_hat so that it has the same shape as F and C_hat
    n_hat = np.r_[0, n_hat]

    theta = sigma, alpha, beta, lamb, gamma

    if plot:
        _make_plots(F + offset, n_hat, C_hat, theta, dt)

    return n_hat, C_hat, LL, theta


def _get_MAP_spikes(F, C_hat, theta, dt, tol=1E-6, maxiter=100, verbosity=0):
    """
    Used internally by deconvolve to compute the maximum a posteriori
    spike train for a given set of fluorescence traces and model parameters.

    See the documentation for deconvolve for the meaning of the
    arguments

    Returns:    n_hat_best, C_hat_best, LL_best

    """
    npix, nt = F.shape

    sigma, alpha, beta, lamb, gamma = theta

    # baseline-subtracted fluorescence
    F_bl =  F - beta[:, None]

    # used for computing the LL and gradient
    scale_var = 1. / (2 * sigma ** 2)
    lD = lamb * dt

    # used for computing the gradient (M.T.dot(LambdaDelta))
    grad_lnprior = np.zeros(nt, dtype=DTYPE)
    grad_lnprior[1:] = lD
    grad_lnprior[:-1] += lD * - gamma

    # initial estimate of spike probabilities (should be strictly non-negative)
    n_hat = C_hat[1:] - gamma * C_hat[:-1]
    # assert not np.any(n_hat < 0), "spike probabilities < 0"

    # (actual - predicted) fluorescence
    D = F_bl - alpha[:, None] * C_hat[None, :]

    # initialize the weight of the barrier term to 1
    z = 1.

    # compute initial posterior log-likelihood of the fluorescence
    LL = _post_LL(n_hat, D, scale_var, lD, z)

    nloop1 = 0
    LL_prev, C_hat_prev = LL, C_hat
    terminate_interior = False

    # in the outer loop we'll progressively reduce the weight of the barrier
    # term and check the interior point termination criteria
    while not terminate_interior:

        s = 1.
        d = 1.
        nloop2 = 0

        # converge for this barrier weight
        while (np.linalg.norm(d) > 5E-2) and (s > 1E-3):

            # compute direction of newton step
            d = _direction(n_hat, D, alpha, sigma, gamma, scale_var,
                            grad_lnprior, z)

            # ensure that s starts sufficiently small to guarantee that n_hat
            # stays positive
            hit = -n_hat / (d[1:] - gamma * d[:-1])
            within_bounds = (hit >= 0)

            if np.any(within_bounds):
                terminate_linesearch = False
                s = min(1., 0.99 * np.min(hit[within_bounds]))
            else:
                # force an early termination at this barrier weight if there is
                # no step size that will keep n_hat >= 0
                terminate_linesearch = True;
                s = 0
                z = 0
                if verbosity >= 2:
                    print ("terminating: no step size will keep n_hat >= 0")

            nloop3 = 0

            # backtracking line search for the largest step size that increases
            # the posterior log-likelihood of the fluorescence
            while not terminate_linesearch:

                # update estimated calcium
                C_hat1 = C_hat + (s * d)

                # update spike probabilities
                n_hat = C_hat1[1:] - gamma * C_hat1[:-1]
                # assert not np.any(n_hat < 0), "spike probabilities < 0"

                # (actual - predicted) fluorescence
                D = F_bl - alpha[:, None] * C_hat1[None, :]

                # compute the new posterior log-likelihood
                LL1 = _post_LL(n_hat, D, scale_var, lD, z)
                # assert not np.any(np.isnan(LL1)), "nan LL"

                # only update C_hat & LL if LL improved
                if LL1 > LL:
                    LL, C_hat = LL1, C_hat1
                    terminate_linesearch = True

                # terminate when the step size is essentially zero but we're
                # still not improving (almost never happens in practice)
                elif s < EPS:
                    if verbosity >= 2:
                        print('terminated linesearch: s < EPS on %i iterations'
                              % nloop3)
                    terminate_linesearch = True

                if verbosity >= 2:
                    print('spikes: iter=%3i, %3i, %3i; z=%6.4f; s=%6.4f;'
                          ' LL=%13.4f' % (nloop1, nloop2, nloop3, z, s, LL))

                # reduce the step size
                s /= 5.
                nloop3 += 1

            nloop2 += 1

        # test for convergence
        delta_LL = np.abs((LL - LL_prev) / LL_prev)

        if (delta_LL < tol):
            terminate_interior = True

        elif z < EPS:
            if verbosity >= 2:
                print 'MAP spike train failed to converge before z -> 0'
            terminate_interior = True

        elif nloop1 > maxiter:
            if verbosity >= 2:
                print 'MAP spike train failed to converge within maxiter'
            terminate_interior = True

        LL_prev, C_hat_prev = LL, C_hat

        # increment the outer loop counter, reduce the barrier weight
        nloop1 += 1
        z /= 10.

    return n_hat, C_hat, LL


def _post_LL(n_hat, D, scale_var, lD, z):

    # barrier term
    with np.errstate(invalid='ignore'):
        barrier = np.log(n_hat).sum()       # this is currently a bottleneck

    # sum of squared (predicted - actual) fluorescence
    ssd = D.ravel().dot(D.ravel())       # fast sum-of-squares

    # weighted posterior log-likelihood of the fluorescence
    LL = -(scale_var * ssd) - (n_hat.sum() / lD) + (z * barrier)

    return LL

def _direction(n_hat, D, alpha, sigma, gamma, scale_var, grad_lnprior, z):

    # gradient
    n_term = np.zeros(D.shape[1])
    n_term[:n_hat.shape[0]] = -gamma / n_hat
    n_term[-n_hat.shape[0]:] += 1. / n_hat
    g = (2 * scale_var * D.T.dot(alpha) - grad_lnprior + z * n_term)

    # main diagonal of the hessian
    n2 = n_hat ** 2
    Hd0 = np.zeros(g.shape[0])
    Hd0[:n_hat.shape[0]] = gamma ** 2 / n2
    Hd0[-n_hat.shape[0]:] += 1 / n2
    Hd0 *= -z
    Hd0 += -alpha.dot(alpha) / sigma ** 2

    # upper/lower diagonals of the hessian
    Hd1 = z * gamma / n2

    # solve the tridiagonal system Hd = -g (we use -g, since we want to
    # *ascend* the LL gradient)
    d = trisolve(Hd1, Hd0, Hd1.copy(), -g, inplace=True)

    return d

def _update_theta(n_hat, C_hat, F, theta, dt, learn_theta):

    sigma, alpha, beta, lamb, gamma = theta
    learn_sigma, learn_alpha, learn_beta, learn_lamb, learn_gamma = learn_theta

    npix, nt = F.shape

    if learn_alpha:

        if learn_beta:
            A = np.vstack((C_hat, np.ones(C_hat.shape[0])))
        else:
            A = C_hat[None, :]

        Y, residuals, rank, s = np.linalg.lstsq(A.T, F.T)

        if learn_beta:
            alpha, beta = Y
        else:
            alpha = Y[0]

    elif learn_beta:
        beta = (F - alpha[:, None] * C_hat[None, :]).sum(1) / nt

    if learn_sigma:
        D = F - (alpha[:, None] * C_hat[None, :] - beta[:, None])
        ssd = D.ravel().dot(D.ravel())      # fast sum-of-squares
        sigma = np.sqrt(ssd / nt)           # RMS error

    if learn_lamb:
        lamb = nt / n_hat.sum()

    if learn_gamma:
        warnings.warn('optimising gamma is not yet supported (ignoring)')

    return (sigma, alpha, beta, lamb, gamma)


def _init_theta(F, dt=0.02, rate=0.5, tau=1.0):

    orig_shape = F.shape
    F = np.atleast_2d(F)
    npix, nt = F.shape

    # K is the correction factor when using the median absolute deviation as a
    # robust estimator of the standard deviation of a normal distribution
    # http://en.wikipedia.org/wiki/Median_absolute_deviation
    K = 1.4785

    med_F = np.median(F, axis=1)

    # noise parameter
    abs_dev = np.abs(F - med_F[:, None])
    sigma = np.median(abs_dev) / K          # scalar

    # amplitude
    alpha = med_F                           # vector

    # we need to ensure that (F - beta[:, None]) is strictly positive
    beta = med_F + (F - med_F[:, None]).min() - EPS

    # rate parameter
    lamb = rate                               # scalar

    # decay parameter (fraction of remaining fluorescence after one time step)
    gamma = np.exp(-dt / tau)               # scalar

    return sigma, alpha, beta, lamb, gamma

def _make_plots(F, n_hat, C_hat, theta, dt):

    sigma, alpha, beta, lamb, gamma = theta
    fig, axes = plt.subplots(2, 1, sharex=True)

    t = np.arange(F.shape[1]) * dt
    F_hat = alpha[:, None] * C_hat[None, :] + beta[:, None]

    axes[0].hold(True)
    axes[0].plot(t, F.sum(0), '-b', label=r'$F$')
    axes[0].plot(t, F_hat.sum(0), '-r', lw=2, label=r'$\alpha\hat{C}+\beta$')
    axes[0].legend(loc=1, fancybox=True, fontsize='large')

    axes[1].plot(t, n_hat, '-k')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel(r'$\hat{n}$', fontsize='large')
    axes[1].set_xlim(0, t[-1])

    return fig, axes

def _detrend(x, dt=0.02, stop_hz=0.01, order=5, plot=False):

    nyquist = 0.5 / dt
    stop = stop_hz / nyquist

    # b, a = signal.cheby2(order, atten, Wn=stop, btype='lowpass')
    b, a = signal.butter(order, Wn=stop, btype='lowpass')

    y = signal.filtfilt(b, a, x, axis=-1)

    if plot:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        ax1.hold(True)
        t = np.arange(x.shape[0]) * dt
        ax1.plot(t, x, '-b')
        ax1.plot(t, y, '-r', lw=2)
        ax2.hold(True)
        ax2.axhline(0, ls='-', color='k')
        ax2.plot(t, x - y, '-b')

    return y

def _boxcar(F, dt=0.02, avg_win=1.0):

    orig_shape = F.shape
    F = np.atleast_2d(F)
    npix, nt = F.shape

    # boxcar filtering
    win_len = max(1, avg_win / dt)
    win = np.ones(win_len) / win_len
    Fsmooth = ndimage.convolve1d(F, win, axis=1, mode='reflect')

    return Fsmooth.reshape(orig_shape)


def _s2h(ss):
    """convert seconds to a pretty "d hh:mm:ss.s" format"""
    mm, ss = divmod(ss, 60)
    hh, mm = divmod(mm, 60)
    dd, hh = divmod(hh, 24)
    tstr = "%02i:%04.1f" % (mm, ss)
    if hh > 0:
        tstr = ("%02i:" % hh) + tstr
    if dd > 0:
        tstr = ("%id " % dd) + tstr
    return tstr
