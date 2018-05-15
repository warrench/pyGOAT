# -*- coding: utf-8 -*-
"""
Created on Mon May  7 09:48:56 2018

@author: Chris
"""

import numpy as np
import sympy as sym
import pyGOAT as goat
import scipy as sp
n = 4
# Define the drift Hamiltonian
def Transmon(n):
    adag = np.zeros((n,n))
    for i in range(n):
        adag[i,i-1] = np.sqrt(i)
    a = adag.T

    def INT_FUNC(Ej,Ec):
        omega = np.sqrt(8*Ej*Ec)
        N = adag.dot(a)       
        
        H = omega*(N) - Ec*N.dot(N)
        return H
    
    return INT_FUNC,a,adag

H, a,adag = Transmon(n)
Hrun = H(15,0.25)
fd = Hrun[1,1]


H0 = sym.Matrix(Hrun)
#Define the drive Hamiltonian
t = sym.symbols('t')
#First with just the carrier
InPhase = adag+a
OutPhase = 1j*(adag-a)
InPhase = sym.cos(2*sym.pi*fd*t)*sym.Matrix(InPhase)
OutPhase = sym.sin(2*sym.pi*fd*t)*sym.Matrix(OutPhase)


Nparams = 3

pulseX = {}
pulseY = {}

for i in range(Nparams):
    pulseX[i] = {sym.symbols('AX'+'_'+str(i)):2*np.random.rand()-1,
                 sym.symbols('tX'+'_'+str(i)):np.random.rand()*10+10,
                 sym.symbols('sigmaX'+'_'+str(i)):np.random.rand()}
    
    pulseY[i] = {sym.symbols('AY'+'_'+str(i)):2*np.random.rand()-1.0,
                 sym.symbols('tY'+'_'+str(i)):np.random.rand()*10+10,
                 sym.symbols('sigmaY'+'_'+str(i)):np.random.rand()}
#Make the pulses
pulselistX = []
pulselistY = []
for i in range(Nparams):
    temp = pulseX[i]
    temp = list(temp.keys())
    pulselistX.append(goat.GaussianPulse(temp[0],
                                          temp[1],
                                          temp[2]))
    temp = pulseY[i]
    temp = list(temp.keys())
    pulselistY.append(goat.GaussianPulse(temp[0],
                                              temp[1],
                                              temp[2]))
pulse1 = goat.ErfComponent(1,0.5,10,20)*goat.superposition(pulselistX)
pulse2 = goat.ErfComponent(1,0.5,10,20)*goat.superposition(pulselistY)

Ht = (2*sym.pi)* (H0 + pulse1*InPhase + pulse2*OutPhase)
dHt = goat.gradient(Ht)

params = {}
for i in range(Nparams):
    temp1 = pulseX[i]
    temp2 = pulseY[i]
    for key in dHt:
        try:
            params[key] = temp1[key]
        except:
            try:
                params[key] = temp2[key]
            except:
                pass

Utarg = np.zeros((n,n))
Utarg[0,1] = 1
Utarg[1,0] = 1
Utemp = np.identity(n)
Utemp[0,0] = 0
Utemp[1,1] = 0

Utarg = Utarg + Utemp

alpha = np.array(list(params.values()))

times = np.linspace(0,30,601)
opts = {'maxiter':250}
print('Starting...')
res = sp.optimize.minimize(goat.minfunc,
                           alpha,
                           method='L-BFGS-B',
                           args=(Ht,dHt,Utarg,times,params),
                           jac=True,
                           options=opts
                           )
print(res.x)






    
    