# pyGOAT

## Installation and Usage

pyGOAT is an open source implementation of the GOAT algorithm developed by Machnes et. al (2018). The original paper can be found here.

https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.120.150401

Important notes:

For pyGOAT to work properly the sympy package needs to be altered slightly to allow for scipy usage in the lambdify function. Sympy does not have a native implementation of the error function which supports arrays. This can be done by making some small changes to the lambdify.py backend.

Find your sympy installation folder and navigate to \sympy\utilities and edit the lambdify.py file to include in the namespace


>SCIPY = {}

>SCIPY_DEFAULT = {"I": 1j}

>SCIPY_TRANSLATIONS = {}

And add the following dictionary entry to the MODULES dictionary

>"scipy": (SCIPY, SCIPY_DEFAULT, SCIPY_TRANSLATIONS, ("from scipy.special import *",))


This package also makes use of the odeintw wrapper to allow odeint to take in matrices as an initial state for computing the time evolved unitary. This can be found here,

https://github.com/WarrenWeckesser/odeintw

