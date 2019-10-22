PyFNND
=======
A Python implementation of fast non-negative deconvolution
------------------------------------

This is a Python implementation of Joshua Vogelstein's [fast non-negative deconvolution](https://github.com/jovo/fast-oopsi/raw/master/fast-oopsi.pdf) algorithm for inferring spikes from neuronal calcium imaging data. Fast non-negative deconvolution uses an interior point method to solve the following optimization problem:

![n^* = \underset{n \geq 0}{\mathrm{argmax}} ~ p(n | F)][1]

where *n** is the maximum a posteriori estimate for the most likely spike train, given the fluorescence signal *F*, and the model:

![C_t &= e^{-\delta t / \tau} \cdot C_{t - 1} + n_t, ~ n &\sim Exp(\lambda \cdot \delta t), \\ F_{p,t} &= \alpha_p \cdot C_t + \beta_p + \epsilon_{p,t}, ~ \epsilon &\sim \mathcal{N}(0, \sigma).][2]

It is also possible to estimate the model parameters σ, α, β, and λ from the data using pseudo-EM updates.

This version was written for the [Kaggle Connectomics Challenge][3], and was optimized for speed when dealing with very large arrays of fluorescence data. In particular, it wraps the [`dgtsv`][4] LAPACK subroutine to efficiently solve for the update direction for each Newton step. The optional dependency on [`joblib`][5] allows multiple fluorescence traces to be processed in parallel.

New in version 0.4
-------------
* Python3 is now supported (thanks to Keji Li)

New in version 0.3
-------------
* Big performance improvements for multi-pixel datasets. By pre-projecting the real fluorescence onto the estimated mask, the interior point algorithm operates only on 1D vectors over time, rather than 2D vectors over pixels and time
* Added an option to decimate the input array over time before initializing or updating the theta parameters. This is particularly useful for large multi-pixel datasets, where a lot of time is spent performing the least-squares fit to update the estimates of the mask and baseline.
* Removed some clutter and improved the clarity of some of the source code.

Dependencies
-------------
* `numpy`
* `scipy`
* `matplotlib`
* `joblib` (optional, required for parallel processing of multiple fluorescence traces)
* A shared LAPACK library (source available [here][6]; Ubuntu users can `$ sudo apt-get install liblapack`)

PyFNND has been tested on machines running Ubuntu Linux (14.04 and 15.10), and using `numpy` >= 1.8.1 and `scipy` >= 0.14.0, as well as the current bleeding-edge dev versions of both libraries. Both Python2 and Python3 are supported. Comments, suggestions and bug reports are all welcome.

Installation
---------------
Install using `pip`:

    $ pip install pyfnnd

Or from the root of the source distribution, simply call

    $ python setup.py install

Example usage
-----------------

```python
from pyfnnd import deconvolve, demo, plotting

# generate a synthetic fluorescence movie
F, C, n, theta = demo.make_fake_movie(1000, dt=0.02, mask_shape=(64, 64),
                                      sigma=0.003, seed=0)

# deconvolve it, learning alpha, beta and lambda
n_best, C_best, LL, theta_best = deconvolve(
    F, dt=0.02, verbosity=1, learn_theta=(0, 1, 1, 1, 0),
    spikes_tol=1E-6, params_tol=1E-6
)

# plot the fit against the true parameters
plotting.ground_truth_2D(F, n_best, C_best, theta_best, n, C, theta, 0.02,
                         64, 64)
```

![Test plot][7]

Reference
----------
Vogelstein, J. T., Packer, A. M., Machado, T. A., Sippy, T., Babadi, B., Yuste, R., & Paninski, L. (2010). Fast nonnegative deconvolution for spike train inference from population calcium imaging. Journal of Neurophysiology, 104(6), 3691-704. [doi:10.1152/jn.01073.2009][8]

License
-----
PyFNND is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

[![No Maintenance Intended](http://unmaintained.tech/badge.svg)](http://unmaintained.tech/)

[1]:http://i.imgur.com/QHrzzvv.png
[2]:http://i.imgur.com/nWgECEC.png
[3]:https://www.kaggle.com/c/connectomics
[4]:http://www.netlib.org/lapack/explore-html/d1/db3/dgtsv_8f.html
[5]:https://pythonhosted.org/joblib
[6]:http://www.netlib.org/lapack/#_software
[7]:http://i.imgur.com/gBGuHBU.png
[8]:http://dx.doi.org/10.1152/jn.01073.2009
