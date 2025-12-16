#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:22:17 2025

Purpose: Landau-de Gennes Modeling
Author:  Rongxing Xu @UESTC, Cuiling Meng's Group
Email: xurongxing123@gmail.com
"""

import numpy as np
from ults_3Dvortex import compute_gradient, pseudo_grad_energy_LdG, pseudo_grad_energy_elastic #For details, please contact the author

def levi_civita_array():
    epsilon = np.zeros((3, 3, 3))
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1
    return epsilon


def kronecker_delta_array(shape=(3,3)):
    return np.eye(*shape)


def compute_S(n):
    nx, ny, nz = n.shape[1], n.shape[2], n.shape[3]
    S = np.zeros((nx, ny, nz))
    
    neighbors = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1)
    ]
    
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                avgcos2 = 0.0
                valid_neighbors = 0
                for dx, dy, dz in neighbors:
                    if (i + dx >= 0) and (i + dx < nx) and \
                       (j + dy >= 0) and (j + dy < ny) and \
                       (k + dz >= 0) and (k + dz < nz):
                        nx_neighbor = n[0, i+dx, j+dy, k+dz]
                        ny_neighbor = n[1, i+dx, j+dy, k+dz]
                        nz_neighbor = n[2, i+dx, j+dy, k+dz]
                        
                        dot = (n[0, i, j, k] * nx_neighbor +
                               n[1, i, j, k] * ny_neighbor +
                               n[2, i, j, k] * nz_neighbor)
                        cos2 = dot ** 2
                        avgcos2 += cos2
                        valid_neighbors += 1

                if valid_neighbors > 0:
                    avgcos2 /= valid_neighbors
                    S[i, j, k] = 0.5 * (3 * avgcos2 - 1) * Sexp
                else:
                    S[i, j, k] = 0.0 
    return S



def initial_director_field(X, Y, Z):
    nx0 = 0*np.ones_like(X)
    ny0 = 1*np.ones_like(X)
    nz0 = 0*np.ones_like(X)

    return np.array([nx0, ny0, nz0])


def dirichlet_boundary_conditions(X, Y, Z):
    Phib = q_tpc * np.arctan2(Y-yc, X-xc) + phi_i
    nxb = np.cos(Phib)
    nyb = np.sin(Phib)
    nzb = 0*np.ones_like(X)
    nb = np.array([nxb, nyb, nzb])
    Q_b = n2Q(nb)

    return Q_b


def n2Q(n):
    S = compute_S(n)
    kronecker_delta = kronecker_delta_array(shape=(3,3))
    Q = S * (np.einsum('i...,j...->ij...', n, n) - \
                1/3 * kronecker_delta[:,:, np.newaxis, np.newaxis, np.newaxis])

    return Q

def energy_LdG(Q):
    fldg1 = 0.5 * A * np.einsum('ij...,ji...->...', Q, Q)
    fldg2 = 1/3 * B * np.einsum('ij...,jk..., ki...->...', Q, Q, Q)
    fldg3 = 1/4 * C * ( 2/3 * np.einsum('ij...,ji..., kl..., lk...->...', Q, Q, Q, Q)\
                       + 1/3 * np.einsum('ij...,jk..., kl..., li...->...', Q, Q, Q, Q) )
    return dv * np.sum(fldg1 + fldg2 + fldg3)




def energy_elastic(Q):
    gradQ = compute_gradient(Q, h)
    epsilon = levi_civita_array()
    f1 = 0.5 * L1 * np.einsum('ijk...,kij...->...', gradQ, gradQ)
    f2 = 0.5 * L2 * np.einsum('ijj...,ikk...->...', gradQ, gradQ)
    f3 = 0.5 * L3 * np.einsum('ikj...,ijk...->...', gradQ, gradQ) #=0
    f4 = 0.5 * L4 * np.einsum('lik,lj...,kij...->...', epsilon, Q, gradQ) #=0
    
    return dv * np.sum(f1 + f2 + f3 + f4)
    



Lx, Ly, Lz = 4, 4, 1 #
Nx, Ny, Nz = 129, 129, 33
h = Lx / (Nx - 1)
xc = Lx/2
yc = Ly/2
zc = Lz/2
dv = h*h*h

q_tpc = -2
phi_i = 0/3*np.pi/2

dp_ratio = 0 #
q0 = dp_ratio * (2*np.pi/Lz) #

K11 = 1
K22 = 1
K33 = 1
K24 = 0

B = -2
C = 2
A = -1/6
Sexp = (-B + np.sqrt(B**2 - 24*A*C))/(4*C)

L1 = (K33 - K11 + 3*K22)/(6 * Sexp**2)
L2 = (K11 - K22 - K24)/(Sexp**2)
L3 = K24/(Sexp**2)
L4 = q0 * K22 * 2/(Sexp**2) 

x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
z = np.linspace(0, Lz, Nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

n0 = initial_director_field(X, Y, Z)
n = n0
Q = n2Q(n)

Qb = dirichlet_boundary_conditions(X, Y, Z)
Q[..., -1] = Qb[..., -1]
Q[..., 0] = Qb[..., 0]


Maxstep = 1201
alpha = 0.25*h*h/K11


for iterstep in range(Maxstep):
    pseudo_grad = pseudo_grad_energy_elastic(Q)+ pseudo_grad_energy_LdG(Q)
    Q = Q - alpha * pseudo_grad
    
    Q[..., -1] = Qb[..., -1]
    Q[..., 0] = Qb[..., 0]























