#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 11:51:13 2025

Purpose: Ray-tracing in 3D liquid crystal film
Author:  Rongxing Xu @UESTC, Cuiling Meng's Group
Email: xurongxing123@gmail.com
"""

import numpy as np
from scipy.integrate import odeint
from ults_Raytracing import InterpolatedQ, GradientQ #For details, please contact the author


def levi_civita_array():
    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1

    return epsilon


def kronecker_delta_array(shape=(3,3)):
    
    return np.eye(*shape)


def Q_loaddata(Nx, Ny, Nz, q_tpc, phi_i):
    "blank, load from data"
    return Q

def compute_S(interp_Q, position):
    Q_value = interp_Q.evaluate(position)
    
    eigenvalues, eigenvectors = np.linalg.eig(Q_value)
    max_eigenvalue_index = np.argmax(eigenvalues)
    
    S_value = eigenvalues[max_eigenvalue_index]*3/2
    return S_value


def canonical_eq(Y, interp_Q, grad_Q):

    px, py, pz, rx, ry, rz = Y
    p = np.array([px, py, pz])
    r = np.array([rx, ry, rz])
    
    Q_value = interp_Q.evaluate(r)
    gradQ_value = grad_Q.evaluate(r)
    S_value = Sexp

    const1 = (2*e_perp + e_para)/(3*e_perp*e_para)
    const2 = e_a/(e_perp*e_para*S_value)
    const3 = e_a/(2*e_perp*e_para*S_value)
    
    drdt = const1 * p + const2 * np.einsum('ij,j->i', Q_value, p)
    dpdt = -const3 * np.einsum('j,k,jki->i', p, p, gradQ_value)
    
    dYdt = np.array([dpdt[0], dpdt[1], dpdt[2],
                     drdt[0], drdt[1], drdt[2]])
    
    return dYdt

def eqfunc(Y, t):
    px, py, pz, rx, ry, rz = Y
    dYdt = canonical_eq(Y, interp_Q, grad_Q)
    return dYdt


def n2Q(n):
    kronecker_delta = kronecker_delta_array(shape=(3,3))
    Q = Sexp * (np.einsum('i...,j...->ij...', n, n) - \
                1/3 * kronecker_delta[:,:, np.newaxis, np.newaxis, np.newaxis])

    return Q

def Q2n(Q):
    n = np.zeros((3, Nx, Ny, Nz))
    S = np.zeros((Nx, Ny, Nz))

    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz):
                Q_ijk = Q[..., i, j, k]
                eigenvalues, eigenvectors_ijk = np.linalg.eig(Q_ijk)
                max_eigenvalue_index = np.argmax(eigenvalues)
                S[i, j, k] = eigenvalues[max_eigenvalue_index]*3/2
                n[..., i, j, k] = eigenvectors_ijk[:, max_eigenvalue_index]
    
    return n, S


Lx, Ly, Lz = 4, 4, 1
Nx, Ny, Nz = 129, 129, 33
h = Lx / (Nx - 1)
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
q_tpc = 2
phi_i = 1/3*np.pi/2
Sexp = 0.5

ne = 1.7
no = 1.5
delta_n = ne - no
e_para = ne**2
e_perp = no**2
e_a = e_para - e_perp


Q = Q_loaddata(Nx, Ny, Nz, q_tpc, phi_i)
n, S = Q2n(Q)
Qnew = n2Q(n)

interp_Q = InterpolatedQ(Qnew, x, y, z)
grad_Q = GradientQ(Qnew, h)

nx, ny, nz = n[0], n[1], n[2]


zi0 = Lz/2
num_rays = 10

ystart = Ly/2 - 10*h
xi = 0 * np.ones(num_rays)
yi = np.linspace(ystart, Ly - ystart, num_rays) 
zi = zi0 * np.ones(num_rays)

thetai = 0 * np.ones(num_rays)
kronecker_delta = kronecker_delta_array(shape=(3,3))

t = np.linspace(0, 4, 4000)

for num in range(num_rays):
    ri = np.array([xi[num], yi[num], zi[num]])
    Qi = interp_Q.evaluate(ri)
    Si = Sexp
    vi = np.array([np.cos(thetai[num]), np.sin(thetai[num]), 0])
    temp_mat = Qi/Si + 1/3 *kronecker_delta
    pi = e_para * vi - e_a * np.einsum('ij, j -> i', temp_mat, vi)
    ini_cond = np.array([pi[0], pi[1], pi[2],
                          ri[0], ri[1], ri[2]])
    sol = odeint(eqfunc, ini_cond, t)














