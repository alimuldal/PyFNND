PyFNND
=======
A Python implementation of fast non-negative deconvolution
------------------------------------

This is a Python implementation of Joshua Vogelstein's [fast non-negative deconvolution](https://github.com/jovo/fast-oopsi/raw/master/fast-oopsi.pdf) algorithm for inferring spikes from neuronal calcium imaging data. Fast non-negative deconvolution uses an interior point method to solve the following optimization problem:

    n_best = argmax_{n >= 0} P(n | F)

where `n_best` is the maximum a posteriori estimate for the most likely spike train, given the fluorescence signal `F`, and the model:

    C_{t} = gamma*C_{t-1} + n_{t},                          n_{t} ~ Exponential(lambda * dt)
    F_{p,t} = alpha_{p}*C_{t} + beta_{p} + epsilon_{p,t},   epsilon ~ N(0, sigma)

It is also possible to estimate the model parameters sigma, alpha, beta and lambda from the data using pseudo-EM updates.

This version was written for the [Kaggle Connectomics Challenge](https://www.kaggle.com/c/connectomics), and was optimized for speed when dealing with very large arrays of fluorescence data. In particular, it wraps the [`dgtsv`](http://www.netlib.org/lapack/explore-html/d1/db3/dgtsv_8f.html) subroutine from the LAPACK Fortran library to efficiently solve for the update direction for each Newton step. The optional dependency on [`joblib`](https://pythonhosted.org/joblib) allows multiple fluorescence traces to be processed in parallel.

New in version 0.2
-------------
* Ability to infer cell pixel masks from movie frames (multiple overlapping masks are not yet supported)
* Improved robustness of parameter estimation using backtracking linesearch

Dependencies
-------------
* `numpy`
* `scipy`
* `matplotlib`
* `joblib` - optional, required for parallel processing
* A shared LAPACK library (source is available from [here](http://www.netlib.org/lapack/#_software); Ubuntu users can simply `$ sudo apt-get install liblapack`)

PyFNND has been tested on machines running Ubuntu Linux (14.10), and using `numpy` v1.8.1 and `scipy` v0.14.0, as well as the current bleeding-edge dev versions of both libraries. Comments, suggestions and bug reports are all welcome.

Installation
---------------
From the root of the source distribution, simply call

    $ python setup.py install

Test plots
-----------------

    from pyfnnd import demo
    demo.make_demo_plots()

![Imgur](http://i.imgur.com/xfJhO0Z.png)

Reference
----------
Vogelstein, J. T., Packer, A. M., Machado, T. a, Sippy, T., Babadi, B., Yuste, R., & Paninski, L. (2010). Fast nonnegative deconvolution for spike train inference from population calcium imaging. Journal of Neurophysiology, 104(6), 3691-704. doi:10.1152/jn.01073.2009

License
-----
PyFNND is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.