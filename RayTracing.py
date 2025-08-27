#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 1 16:11:10 2024

Author: Rongxing Xu @uestc
Purpose: Modeling ray tracing in 2d vortical LC


"""

import numpy as np
from scipy.integrate import odeint
from ults.ults_RT import christoffel_symbols

def vortex_field(pos):
    xx, yy = pos
    theta = k * np.arctan2(yy, xx) + c  
    n_x = np.cos(theta)  
    n_y = np.sin(theta) 
    return n_x, n_y

def geodesic_eq(vety, pos):
    n = vortex_field(pos)
    gamx_xx, gamy_xx, gamx_xy, gamy_xy, gamx_yy, gamy_yy = christoffel_symbols(pos, n)
    vx, vy = vety
    ax = -(gamx_xx * vx**2 + 2 * gamx_xy * vx * vy + gamx_yy * vy**2)
    ay = -(gamy_xx * vx**2 + 2 * gamy_xy * vx * vy + gamy_yy * vy**2)
    return np.array([ax, ay])

def eqfunc(zz, t):
    xx, yy, vx, vy = zz
    aa = geodesic_eq([vx, vy], [xx, yy])
    return [vx, vy, aa[0], aa[1]]

# Constants
k = -1
c = 0
ne = 1.7
no = 1.5
delta_n2 = ne**2 - no**2
delta_n = ne-no
yi = np.linspace(-0.2, 0.2, 5) 
xi = -20 * np.ones_like(yi)
vxi = np.ones(len(xi)) 
vyi = np.zeros(len(xi))

x = np.linspace(-30, 30, 600)
y = np.linspace(-30, 30, 600)
t = np.linspace(0, 50, 5000)
for num in range(11):
    ini_cond = [xi[num], yi[num], vxi[num], vyi[num]]
    sol = odeint(eqfunc, ini_cond, t)

