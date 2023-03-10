#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 12 17:39:17 2022

@author: Brice
"""

## Importations ###############################################################

import numpy as np
from scipy.fft import fftshift, ifftshift, fft, ifft
import matplotlib.pyplot as plt

## Some parameters of the problem #############################################

X = np.arange(-45, 45, 0.01)                        # Space domain
a = 2                                               # Try with 1
m = 1                                               # Try with 5
Phi = (1/(2*np.pi)**0.25/a**0.5)*np.exp(-X*X/4/a/a) # φ such that ∫|φ|^2 = 1
b = -1                                              # Try with -1
Phib = np.exp(1j*b*X)*Phi
Phibb = np.cos(b*X)*Phi
dx = X[1]-X[0]                                      # Space step

## Fonction for the solution ##################################################

def myfft(X):
    return fftshift(fft(X))                         # Shifted fft

def myifft(X):
    return ifft(ifftshift(X))                       # Shifted ifft

def Psi(t):
    return myifft(np.exp(-1j*X*X*t/2/m)*myfft(Phi))

# The resolution does not work without the shift !

## Computation of the solution ################################################

T = np.arange(0, 3000, 100)

for i,t in enumerate(T):
    Sol  = Psi(t)
    Absv = Sol.real**2+Sol.imag**2                  # Absolute value ^2 of sol
    print(sum(Absv*dx))                             # Control of the integral
    mean = sum(X*Absv*dx)                           # Mean position
    plt.plot(X, Absv, label = r"$|\Psi(t,x)|^2$")   # Plot of the solution
    plt.plot(mean, 0, "o", label = "Mean position") # Plot of mean position
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim([0, 0.3])
    plt.title("m = {0:.0f}\nt = {1:.2f}".format(m, t))
    plt.legend()
    plt.savefig("pic_"+str(i)+".pdf")               # To save the figure
    plt.clf()                                       # Clear current fig

# To get a movie:
# First, run this file !
# Then, do this in a linux terminal:
# $ ls pic_* | sort -n -t _ -k 2 > tmp.txt
# $ convert -delay 10 -loop 0 -density 300 $(cat tmp.txt) film.mp4
# $ rm pic_* tmp.txt
