# -*- coding: utf-8 -*-
"""
Created on Thu May  3 17:40:08 2018

@author: Chris
"""

#pygoat test1

import sympy as sym
import numpy as np
import scipy as sp
import pyGOAT as goat

#======= Whole process ========

# Create the pulses with some sympy parameters
# Pull out all the paramters and their values
# sequence.freesymbols
# remove 't' symbol
t = sym.symbols('t')
X = sym.Matrix(np.array([[0,1],[1,0]]))
Y = sym.Matrix(np.array([[0,-1j],[1j,0]]))
Z = 0.5*sym.Matrix(np.array([[1,0],[0,-1]]))
X[0,1] = sym.E**(-sym.I*2*sym.pi*t)
X[1,0] = sym.E**(sym.I*2*sym.pi*t)
Y[0,1] = Y[0,1]*sym.E**(-sym.I*2*sym.pi*t)
Y[1,0] = Y[1,0]*sym.E**(sym.I*2*sym.pi*t)
#    
# Make the pulse parameters for X and Y
# We will use 3 Gaussian control pulses each for X and Y
N = 3
#    
pulseX = {}
pulseY = {}
for i in range(N):
    pulseX[i] = {sym.symbols('AX'+'_'+str(i)):2*np.random.rand()-1,
                 sym.symbols('tX'+'_'+str(i)):np.random.rand()*10+10,
                 sym.symbols('sigmaX'+'_'+str(i)):np.random.rand()}
    
    pulseY[i] = {sym.symbols('AY'+'_'+str(i)):2*np.random.rand()-1.0,
                 sym.symbols('tY'+'_'+str(i)):np.random.rand()*10+10,
                 sym.symbols('sigmaY'+'_'+str(i)):np.random.rand()}
# Make the pulses
pulselistX = []
pulselistY = []
for i in range(N):
    temp = pulseX[i]
    temp = list(temp.keys())
    pulselistX.append(goat.GaussianPulse(temp[0],temp[1],temp[2]))
    temp = pulseY[i]
    temp = list(temp.keys())
    pulselistY.append(goat.GaussianPulse(temp[0],temp[1],temp[2]))
pulse1 = goat.ErfComponent(1,0.5,10,30)*goat.superposition(pulselistX)
pulse2 = goat.ErfComponent(1,0.5,10,30)*goat.superposition(pulselistY)
#Make Hamiltonian    

H = (2*sym.pi)*(Z + pulse1*X + pulse2*Y)
dH = goat.gradient(H)

#     Make a dictionary of params in the order of dH
params = {}
for i in range(N):
    temp1 = pulseX[i]
    temp2 = pulseY[i]
    for key in dH:
        try:
            params[key] = temp1[key]
        except:
            try:
                params[key] = temp2[key]
            except:
                pass

#
#Make the target unitary
Utarg = (1/np.sqrt(2))*np.array([[1,1],[1,-1]])
#    Utarg = np.array([[0,1],[1,0]])
#Make the alpha vector to pass to the optimizer
alpha = np.array(list(params.values()))
#How long to run
times = np.linspace(0,40,401)
opts = {'maxiter':250}
res = sp.optimize.minimize(goat.minfunc, 
                           alpha, 
                           method='L-BFGS-B',
                           args=(H,dH,Utarg,times,params),
                           options=opts
                           )
print(res.x)