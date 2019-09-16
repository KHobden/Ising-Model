#4D Ising Model
#Kieran Hobden
#05-Mar-'19

import numpy as np
import math
import matplotlib.pyplot as plt
import time

"""Start time"""
start = time.clock()

"""Create an nxnxnxn grid"""
n = 4
grid = np.ones((n, n, n, n))

"""Define constants"""
J = 1
mu = 1
H = 1
k_b = 1

"""Define number of times to analyse flip of every component"""
N = 10

"""Compute the sum over neighbours for use in the energy calculation and to
increase efficiency """
def neighbour_sum(i, j, k, l):
    sum = grid[i][j][k][l]*(grid[(i+1)%n][j%n][k%n][l%n] + grid[(i-1)%n][j%n][k%n][l%n] + \
    grid[i%n][(j+1)%n][k%n][l%n] + grid[i%n][(j-1)%n][k%n][l%n] + \
    grid[i%n][j%n][(k+1)%n][l%n] + grid[i%n][j%n][(k-1)%n][l%n] + \
    grid[i%n][j%n][k%n][(l+1)%n] + grid[i%n][j%n][k%n][(l-1)%n])
    return sum

"""Compute the energy of the grid"""
def energy():
    int_term = 0.0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for l in range(n):
                    int_term += neighbour_sum(i, j, k, l)
    spin_sum = np.sum(grid)
    energy = -(J/2)*int_term - mu*H*spin_sum
    return energy


"""Flipping algorithm"""
x_val = np.arange(N*n**4)
def flipping(T):
    y_val = np.zeros(N*n**4)
    global grid
    grid = np.ones((n, n, n, n))
    for a in range(N):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        e_a = energy()
                        y_val[a*n**4 + i*n**3 + j*n**2 + k*n + l] = e_a
                        grid[i][j][k][l] = -grid[i][j][k][l]
                        e_b = e_a + 2*neighbour_sum(i, j, k, l)
                        if e_b > e_a and np.random.rand() < math.exp((e_a - e_b)/(k_b*T)):
                            grid[i][j][k][l] = -grid[i][j][k][l]
    return y_val

"""Plot energy with time"""
plt.plot(x_val, flipping(10), "b", label="T=10")
plt.plot(x_val, flipping(30), "g", label="T=30")
plt.plot(x_val, flipping(100), "r", label="T=100")
plt.title("4D Ising Model")
plt.xlabel("No. of iterations")
plt.ylabel("Energy")
plt.legend(loc="best")

"""Stop time"""
print("Time taken by code:", time.clock() - start, "seconds")

"""Plot"""
plt.show()
