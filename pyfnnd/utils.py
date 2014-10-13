import numpy as np
from scipy import ndimage, signal


def detrend(x, dt=0.02, stop_hz=0.01, order=5):
    orig_shape = x.shape
    x = np.atleast_2d(x)
    nyquist = 0.5 / dt
    stop = stop_hz / nyquist
    b, a = signal.butter(order, Wn=stop, btype='lowpass')
    y = signal.filtfilt(b, a, x, axis=1)
    return (x - y).reshape(orig_shape)

def boxcar(F, dt=0.02, avg_win=1.0):
    orig_shape = F.shape
    F = np.atleast_2d(F)
    npix, nt = F.shape
    win_len = max(1, avg_win / dt)
    win = np.ones(win_len) / win_len
    Fsmooth = ndimage.convolve1d(F, win, axis=1, mode='reflect')
    return Fsmooth.reshape(orig_shape)


def s2h(ss):
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
