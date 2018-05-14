# -*- coding: utf-8 -*-
"""
Created on Fri May  4 16:44:11 2018

@author: Chris
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sympy as sym
import pyGOAT as goat

#Make the matrices    

times = np.linspace(0,30,601)
t = sym.symbols('t')

window = goat.ErfComponent(1,0.5,5,25)

y = sym.lambdify(t,window,modules=['numpy',
                                   {'erf': sp.special.erf},
                                   {'erfc': sp.special.erfc}])

y1 = y(times)
plt.plot(times,y1)

