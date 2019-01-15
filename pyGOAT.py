# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 15:48:08 2018

@author: Chris
"""

import sympy as sym
from odeintw import odeintw
import numpy as np
import scipy as sp
import scipy.special as spec
import sympy.physics.quantum.matrixutils as utils

MODULES = ['numpy',
           {'erf': spec.erf},
           {'erfc': spec.erfc}]


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
    Coupled differential equation to solve. Use sparse*dense to speed up
    """
    RHS = []
    dUdt = -1j*utils.to_scipy_sparse(H(t))*np.matrix(U[0])
#    dUdt = dUdt.toarray()
    RHS.append(dUdt)
    for i,key in enumerate(dH):
        temp = -1j*(utils.to_scipy_sparse(dH[key](t))*np.matrix(U[0]) 
                       + utils.to_scipy_sparse(H(t))*np.matrix(U[i+1]))
        RHS.append(temp)
    return RHS

def integrator(sys,t,H,dH):
    U0 = []
    shape = H(0).shape
    U0.append(np.eye(shape[0],dtype='complex128'))
    for i in range(len(dH)):
        U0.append(1j*np.zeros(shape))
    sol = odeintw(sys,U0,t,args=(H,dH), atol=1e-12, rtol=1e-12)
    U_f = sol[-1]
    print('Integration Complete...')
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
    d = len(Utarg)
    Utarg = Utarg.conj().T
    Utarg = utils.to_scipy_sparse(Utarg)
    g = 1.0 - (1/d)*np.abs(np.trace(Utarg*np.matrix(Uact)))
    return g

def infidelity_jac(Utarg,U):
    d = Utarg.ndim
    #conjugate transpose of Utarg
    Utarg = Utarg.conj().T
    Utarg = utils.to_scipy_sparse(Utarg)
    #Sort out the components of U
    Uact = U[0]
    Ugrad = U[1:]
    #compute the overlap
#    gstar = np.conjugate(np.trace(Utarg.dot(Uact)))
    gstar = np.conjugate(np.trace(Utarg*np.matrix(Uact)))    
    dg = []
    for dU in Ugrad:
        inside = (gstar/np.abs(gstar))*(1/d)*np.trace(Utarg*np.matrix(dU))
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

    #alpha and dH needs to be in the key order
    params = dict(zip(keys,alpha))
    #substitute in the alpha parameters
    Ht = set_params(H,params)
    for key in params:
        dH[key] = set_params(dH[key],params)
    #lambdify the statements
    Ht = sym.lambdify(t,Ht,modules=MODULES)
    dHt = {}
    for key in params:
        dHt[key] = sym.lambdify(t,dH[key],modules=MODULES)
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
    dg = infidelity_jac(Utarg,Uf) 
    
    return g, dg
    
    

    
    
    
    
    
    
    
    
    
    

