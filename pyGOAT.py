# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:48:08 2018

@author: Chris
"""

import sympy as sym
from odeintw import odeintw
import numpy as np
import scipy as sp


#============================= Pulse Types ====================================

def GaussianPulse(A,tau,sigma):
    """ 
    GaussianPulse creates a sympy expression of a Gaussian
    
    Input:
        A: A sympy symbol corresponding to the amplitude
        tau: A sympy symbol corresponding to the offset
        sigma: A sympy symbol corresponding to the std dev
        
    Output:
        A sympy expression corresponding to the Gaussian pulse
    """
    t = sym.symbols('t')
    expr = A*sym.E**(-(t-tau)**2/sigma**2)
    return expr

def FourierComponent(A,f,phi):
    """
    FourierComponent generates a sympy expression for a fourier component
    
    Input:
        A:
        f:
        phi:
    
    Output:
        A sympy expression corresponding to the Fourier component
    """
    t = sym.symbols('t')
    expr = A*sym.sin(2*sym.pi*f*t + phi)
    return expr

def ErfComponent(A,s,t1,t2):
    """
    Return an error function pulse
    
    Input
        A:
        s:
        t1:
        t2:
    Output:
        A sympy expression corresponding to an error function window
    """
    t = sym.symbols('t')
    arg1 = sym.sqrt(sym.pi)*(s/A)*(t-t1)
    arg2 = sym.sqrt(sym.pi)*(s/A)*(t-t2)
    expr = (A/4.0)*(1+sym.erf(arg1))*sym.erfc(arg2)
    return expr

#========================== Operations on Pulses ==============================

def superposition(exprlist):
    """
    Add together all of the pulses in superposition
    
    Input
        exprlist: A list of all the expressions one wants in the output wave
    Output
        expr: A sympy expression of the superposition of all pulses
    """
    expr = 0
    for exp in exprlist:
        expr += exp
    return expr

def gradient(expr):
    """
    Compute the gradient of a sympy expression across all symbols (except t)
    
    Input
        expr: Sympy expression of the input wave
    Output
        gradients: A dictionary of all the gradients of the pulse
    """
    gradients = {}
    symbs = get_params(expr)
    try:
        symbs.remove(sym.symbols('t'))
    except:
        pass
    for symb in symbs:
        gradients[symb] = sym.diff(expr,symb)
    return gradients

def filterfunction(w,expr):
    """
    Apply a window filter function to a sympy expression
    
    Input
        w: window sympy expression
        expr: a sympy expression
    Output
        The windowed expression
    """
    return w*expr

#================= Parameter Manipulation of Pulses ===========================

def get_params(expr):
    """
    Extract the parameters from a sympy expression
    
    Input
        expr: A sympy expression
    Output
        symbs: A python 'set' of symbols in the sympy expression
    """
    symbs = expr.free_symbols
    return symbs

def set_params(pulse,params):
    """
    Substitute the parameters for numerical values of the pulses
    
    
    Input
        pulse: A sympy expression corresponding to the pulse
        params: A dictionary corresponding to all the parameters one wants to
                change
    Output
        expr: A sympy expression with the substituted numerical values
              which can then be lambdified after applying to the matrix
    """
    expr = pulse
    symbs = get_params(expr)
    try:
        symbs.remove(sym.symbols('t'))
    except:
        pass
    #Convert from a set to a list to interate through
    symbs = list(symbs)
    for symb in symbs:
        try:
            expr = expr.subs(symb, params[symb])
        except:
            print(str(symb) + ' is not in the parameter dictionary')
            pass
    return expr

#================= Matrix Manipulation ========================================
    
def PulseMatrixCombo(ctrl,Hctrl):
    """
    Combine a sympy expression with a sympy representation of a matrix to be
    able to labdify it properly
    
    Input
        ctrl: Sympy expression to apply to the control Hamiltonian
        Hctrl: The control Hamiltonian (sympy matrix or numpy array)
    Output
        H_of_t: The functionalized version of the time dependent matrix
                Hctrl(t)
    """
    if isinstance(Hctrl,np.ndarray):
        Hctrl = sym.matrix(Hctrl)
    elif isinstance(Hctrl, sym.matrices.dense.MutableDenseMatrix):
        pass
    else:
        raise TypeError('Hctrl must be either a numpy array or a sympy matrix')
    H_of_t = ctrl*Hctrl
    return H_of_t

def gradient_ctrls(grad,Hctrl):
    """
    Combine the sympy expression for the gradients of the analytic control with
    the matrix representation of the control
    
    Input
        grad: The gradient of the analytic control
        Hctrl: Control Hamiltonian (sympy matrix or numpy array)
    Output
        dH: A dictionary corresponding to all the gradient controls
    """
    
    if isinstance(Hctrl,np.ndarray):       
        Hctrl = sym.Matrix(Hctrl)
    elif isinstance(Hctrl,sym.Matrix):
        pass
    else:
        raise TypeError('Hctrl must be either a numpy array or a sympy matrix')
    dH = {}
    for key in grad:
        dH[key] = grad[key]*Hctrl
    return dH

#============================= Time Evolution =================================
# Need to test
def asys(U,t,H,dH):
    """
    Coupled differential equation to solve
    """
    RHS = []
    dUdt = -1j*H(t).dot(U[0])
    RHS.append(dUdt)
#    for i,key in enumerate(dH):
#        RHS.append(-1j*(dH[key](t).dot(U[0]) + H(t).dot(U[i])))
    return RHS

def integrator(sys,t,H,dH):
    U0 = []
    shape = H(0).shape
    U0.append(np.eye(shape[0],dtype='complex128'))
#    for i in range(len(dH)):
#        U0.append(1j*np.zeros(shape))
    sol = odeintw(sys,U0,t,args=(H,dH), atol=1e-12, rtol=1e-12)
    U_f = sol[-1]
    return U_f

#EVERYTHING IS AWESOME!!!!!!!!
    
#============================ Clean Up Arrays =================================
    
def tidyup(A,atol=1e-6):
    Ar = A.real
    Aim = A.imag

    Ar_bool = np.abs(Ar) >= atol
    Aim_bool = np.abs(Aim) >= atol

    Ar = Ar*Ar_bool
    Aim = 1j*Aim*Aim_bool

    return Ar+Aim


#========================= Gradient Optimization ==============================
#Need to implement

def infidelity(Utarg,Uact):
    d = Utarg.ndim
    Utarg = Utarg.conj().T

    g = 1.0 - (1/d)*np.abs(np.trace(Utarg.dot(Uact)))
    return g

def infidelity_jac(Utarg,U):
    d = Utarg.ndim
    #conjugate transpose of Utarg
    Utarg = Utarg.conj().T
    #Sort out the components of U
    Uact = U[0]
    Ugrad = U[1:]
    #compute the overlap
    gstar = np.conjugate(np.trace(Utarg.dot(Uact)))    
    dg = []
    for dU in Ugrad:
        inside = (gstar/np.abs(gstar))*(1/d)*np.trace(Utarg.dot(dU))
        dg.append(-inside.real)
    dg = np.array(dg)
    return dg
    
    
def minfunc(alpha, H, dH, Utarg, times,params):
    """
    The minimizer objective function takes in the parameter vector alpha,
    the time dependent Hamiltonian, H(t) = H0 + Hctrls(t), and the gradient
    of the time dependent Hamiltonian.
    
    The function will substitute the parameters into H(t) and dH, lambdify the
    expression and compute the time evolution to obtain U(T) and dU(T).
    
    These values will then be used to compute the infidelity g and the gradient
    
    The infidelity and its gradient will then be returned to be applied to the
    optimizer where the new parameters will be applied
    
    alpha is just a dummy to pass into scipy minimize
    """
    #Create a time symbol to symbolize over
    t = sym.symbols('t')
    #This will story the list of keys in order
#    print(alpha)
    keys = list(params.keys())
    #alpha needs to be in the key order
    params = dict(zip(keys,alpha))
    #substitute in the alpha parameters
    Ht = set_params(H,params)
    for key in params:
        dH[key] = set_params(dH[key],params)
    #lambdify the statements
    Ht = sym.lambdify(t,Ht,modules=['numpy','scipy'])
    dHt = {}
    for key in dH:
        dHt[key] = sym.lambdify(t,dH[key],modules=['scipy','numpy'])
    # Run the time evolution
    Uf = integrator(asys,times,Ht,dHt)
    Uf[0] = tidyup(Uf[0])
    
    g = infidelity(Utarg,Uf[0])
    print('The infidelity is:')
    print(g)
    print('\n')
    print('The parameters are:')
    print(params)
    print('\n')
#    dg = infidelity_jac(Utarg,Uf) 
    
    return g#, #dg



if __name__=='__main__':
#    #Make the matrices    
#    Z = np.array([[-1,0],[0,1]])
#    X = np.array([[0,1],[1,0]])
#    Z = sym.Matrix(Z)
#    X = sym.Matrix(X)
#
#    #Make pulse params
#    t = sym.symbols('t')
#    A = sym.symbols('A')
#    t0 = sym.symbols('t0')
#    sigma = sym.symbols('sigma')
#    params = {A:1,t0:15,sigma:1}
#    #Make the pulse and gradient
#    Gauss = GaussianPulse(A,t0,sigma)
#    grad = gradient(Gauss)
#    #Substitute the paramters for the pulse and the gradient
#    Gauss = set_params(Gauss,params)
#    gradsub = {}
#    for key in grad:
#        gradsub[key] = set_params(grad[key],params)
#    #Make the Hamiltonians
#    H = Z + Gauss*X
#    dH = gradient_ctrls(gradsub,X)
##    print(H)
##    print(dH)
#    #Lambdify the matrices
#    H = sym.lambdify(t,H)
#    for key in dH:
#        dH[key] = sym.lambdify(t,dH[key])
#    #integrate the system
#    time = np.linspace(0,30,3001)
#    U_f = integrator(asys,time,H,dH)
#    U_f = U_f[0]
##    print(U_f)
#    U_f = tidyup(U_f)
#    print(U_f)
##    Check it is unitary
#    print(U_f.dot(U_f.conj().T))
    
    
    
    
    
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
        pulselistX.append(GaussianPulse(temp[0],temp[1],temp[2]))
        temp = pulseY[i]
        temp = list(temp.keys())
        pulselistY.append(GaussianPulse(temp[0],temp[1],temp[2]))
    pulse1 = ErfComponent(1,0.5,10,30)*superposition(pulselistX)
    pulse2 = ErfComponent(1,0.5,10,30)*superposition(pulselistY)
    #Make Hamiltonian    
    
    H = (2*sym.pi)*(Z + pulse1*X + pulse2*Y)
    dH = gradient(H)
    
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
    res = sp.optimize.minimize(minfunc, 
                               alpha, 
                               method='L-BFGS-B',
                               args=(H,dH,Utarg,times,params),
                               options=opts
                               )
    print(res.x)
    
    
    

    
    
    
    
    
    
    
    
    
    

