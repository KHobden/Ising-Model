#3D Ising Model
#Kieran Hobden
#02-Mar-'19

import numpy as np
import math
import matplotlib.pyplot as plt
import time

"""Start time"""
start = time.clock()

"""Create an nxnxn grid"""
n = 5
grid = np.ones((n, n, n))

"""Define constants"""
J = 1
mu = 1
H = 1
k_b = 1

"""Define number of times to analyse flip of every component"""
N = 50

"""Compute the sum over neighbours for use in the energy calculation and to
increase efficiency """
def neighbour_sum(i, j, k):
    sum = grid[i][j][k]*(grid[(i+1)%n][j%n][k%n] + grid[(i-1)%n][j%n][k%n] + \
    grid[i%n][(j+1)%n][k%n] + grid[i%n][(j-1)%n][k%n] + \
    grid[i%n][j%n][(k+1)%n] + grid[i%n][j%n][(k-1)%n])
    return sum

"""Compute the energy of the grid"""
def energy():
    int_term = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                int_term += neighbour_sum(i, j, k)
    spin_sum = np.sum(grid)
    energy = -(J/2)*int_term - mu*H*spin_sum
    return energy


"""Flipping algorithm"""
x_val = np.arange(N*n**3)
def flipping(T):
    y_val = np.zeros(N*n**3)
    global grid
    grid = np.ones((n, n, n))
    for a in range(N):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    e_a = energy()
                    y_val[a*n**3 + i*n**2 + j*n + k] = e_a
                    grid[i][j][k] = -grid[i][j][k]
                    e_b = e_a + 2*neighbour_sum(i, j, k)
                    if e_b > e_a and np.random.rand() < math.exp((e_a - e_b)/(k_b*T)):
                        grid[i][j][k] = -grid[i][j][k]
    return y_val

"""Plot energy with time"""
plt.plot(x_val, flipping(10), "b", label="T=10")
plt.plot(x_val, flipping(30), "g", label="T=30")
plt.plot(x_val, flipping(100), "r", label="T=100")
plt.title("3D Ising Model")
plt.xlabel("No. of iterations")
plt.ylabel("Energy")
plt.legend(loc="best")

"""Stop time"""
print("Time taken by code:", time.clock() - start, "seconds")

"""Plot"""
plt.show()
