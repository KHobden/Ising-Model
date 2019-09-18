#Ising Model

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import scipy.ndimage as sc
from scipy.signal import savgol_filter
from scipy.optimize import curve_fit as fit
import math
import random
import time

"""Start time"""
start = time.clock()

"""Choose which tasks to run by setting the task variable to true"""
core_task = False
task_1 = True
task_2 = False
task_3 = False
task_4 = False
task_5 = False

"""Create an nxn grid"""
n = 24

"""Initiate random number generator"""
#random.seed(30)

"""Define constants"""
J = 1
mu = 1
k_b = 1

"""Define number of times to analyse flip of every component"""
N = 200




"""Setup the experiment"""

"""Compute the sum over neighbours for use in the energy calculation and to
increase efficiency """
def calc_neighbour_sum(grid, i, j):
    sum = grid[i][j]*(grid[(i+1)%n][j%n] + grid[(i-1)%n][j%n] + \
    grid[i%n][(j+1)%n] + grid[i%n][(j-1)%n])
    return sum

"""Compute the energy of the grid"""
def energy(grid, H):
    int_term = 0.0
    for i in range(n):
        for j in range(n):
            neighbour_sum = calc_neighbour_sum(grid, i, j)
            int_term += neighbour_sum
    spin_sum = np.sum(grid)
    energy = -(J/2)*int_term - mu*H*spin_sum
    return energy

"""Sweeping flipping algorithm"""
x_val = np.arange(N*n**2)
def flipping(T, H):
    magnetism = np.zeros(N*n**2)
    y_val = np.zeros(N*n**2)
    grid = np.ones((n, n))
    for k in range(N):
        for i in range(n):
            for j in range(n):
                e_a = energy(grid, H)
                magnetism[k*n**2 + i*n + j] = np.sum(grid)
                y_val[k*n**2 + i*n + j] = e_a
                grid[i][j] = -grid[i][j]
                e_b = energy(grid, H)
                if e_b > e_a and random.random() > math.exp((e_a - e_b)/(k_b*T)):
                    grid[i][j] = -grid[i][j]
    return y_val, magnetism

"""Random flipping algorithm"""
def rand_flipping(T, H):
    magnetism = []
    y_val = []
    grid = np.ones((n, n))
    coordination_numbers = []
    for i in range(N*n**2):
        e_a = energy(grid, H)
        magnetism.append(np.sum(grid))
        y_val.append(e_a)
        x = random.randint(0,n-1)
        y = random.randint(0,n-1)
        grid[x][y] = -grid[x][y]
        e_b = energy(grid, H)
        if e_b > e_a and random.random() > math.exp((e_a - e_b)/(k_b*T)):
            grid[x][y] = -grid[x][y]

        #Adds 1 to the coordination number for each similar neighbouring element
        coordination_number = 0
        if grid[x][y] == grid[(x+1)%n][y]:
            coordination_number += 1
        if grid[x][y] == grid[(x-1)%n][y]:
            coordination_number += 1
        if grid[x][y] == grid[x][(y+1)%n]:
            coordination_number += 1
        if grid[x][y] == grid[x][(y-1)%n]:
            coordination_number += 1
        coordination_numbers.append(coordination_number)

    #Returns a domain_size = 1 if all neighbours are the same
    domain_size = 0
    for i in coordination_numbers:
        if i == 4:
            domain_size += 1
        
    return y_val, magnetism, grid, domain_size




"""Core Task"""
if core_task == True:

    """Plot energy against number of iterations"""
    plt.plot(x_val, flipping(10, 0)[0], "r", label="T=10 (sweep)")
    plt.plot(x_val, rand_flipping(10, 0)[0], "b", label="T=10 (random)")
    #plt.plot(x_val, rand_flipping(10, 0)[0], "r", label="T=10 (Above critical temperature)")
    plt.title("2D Ising Model")
    plt.xlabel("No. of iterations")
    plt.ylabel("Energy")
    #plt.ylim(-50, 220)
    plt.legend(loc="best")
    plt.show()


"""Task 1"""
if task_1 == True:

    """Plot magnetisation with time using the random and sweeping algorithms"""
    #plt.plot(flipping(2, 0)[1], "r", label="T=2")
    """
    plt.plot(rand_flipping(2, 0)[1], "b", label="T=2 (Below critical temperature)")
    plt.plot(rand_flipping(10, 0)[1], "r", label="T=10 (Above critical temperature)")
    plt.title("Magnetism against iterations")
    plt.xlabel("No. of iterations")
    plt.ylabel("Magnetism")
    plt.legend(loc="best")
    plt.show()
    """

    """Run model to store the result"""
    magnetism = flipping(10, 0)[1]
    rand_magnetism = rand_flipping(10, 0)[1]
    
    """Pre-made auto-correlation function"""
    average_magnetism = np.average(magnetism[2000:])
    average_rand_magnetism = np.average(rand_magnetism[2000:])
    print("Temperature = 2.6")
    print("Sweeping average: ", average_magnetism)
    print("Random average: ", average_rand_magnetism)

    """Acorr function implementation and plot for sweep and random"""
    M_prime = magnetism[2000:] - average_magnetism
    rand_M_prime = rand_magnetism[2000:] - average_rand_magnetism
    """
    top_left_graph = plt.subplot(221)
    top_left_graph.acorr(M_prime, maxlags=len(M_prime)-1)
    top_left_graph.set_title("Matplotlib")
    
    top_right_graph = plt.subplot(222)
    top_right_graph.acorr(rand_M_prime, maxlags=len(rand_M_prime)-1)
    top_right_graph.set_title("Matplotlib (random)")
    """
    """Numpy autocorrelation function and plot for sweep and random"""
    #autocorr = np.correlate(M_prime, M_prime, mode='full')
    #autocorr /= autocorr.max() #normalises the autocorrelation
    rand_autocorr = np.correlate(rand_M_prime, rand_M_prime, mode='full')
    rand_autocorr /= rand_autocorr.max() #normalises the autocorrelation
    """
    bottom_left_graph = plt.subplot(223)
    bottom_left_graph.plot(np.arange(-N*n**2+2001, N*n**2-2000), autocorr)
    bottom_left_graph.set_title("Numpy")
    
    bottom_right_graph = plt.subplot(224)
    bottom_right_graph.plot(np.arange(-N*n**2+2001, N*n**2-2000), rand_autocorr)
    bottom_right_graph.set_title("Numpy (random)")
    
    plt.tight_layout()
    plt.show()
    """

    """Find tau_e by shifting autocorr by 1/e and then minimising the function"""
    rand_autocorr = np.asarray(rand_autocorr)
    shifted_autocorr = np.abs(rand_autocorr - math.exp(-1))
    #Note the x-axis has zero on the left so the index needs to be adjusted
    tau = np.abs(shifted_autocorr.argmin() - (N*n**2 - 2000))
    print("Tau_e = ", tau)

    """Find values of tau for different temperatures"""
    """
    temperatures_t1 = np.linspace(1, 5, num=500)
    taus = []
    for T in temperatures_t1:
        rand_magnetism2 = rand_flipping(T, 0)[1]
        average_rand_magnetism2 = np.average(rand_magnetism2[2000:])
        rand_M_prime2 = rand_magnetism2[2000:] - average_rand_magnetism2
        rand_autocorr2 = np.correlate(rand_M_prime2, rand_M_prime2, mode='full')
        rand_autocorr2 /= rand_autocorr2.max()
        rand_autocorr2 = np.asarray(rand_autocorr2)
        #Find tau_e by shifting autocorr by 1/e and then minimising the function
        shifted_autocorr2 = np.abs(rand_autocorr2 - math.exp(-1))
        #Note the x-axis has zero on the left so the index needs to be adjusted
        tau2 = np.abs(shifted_autocorr2.argmin() - (N*n**2 - 2000))
        taus.append(tau2)
    plt.plot(temperatures_t1, taus)
    plt.xlabel("Temperature")
    plt.ylabel("Time lag")
    plt.show()
    """

"""Task 2"""
if task_2 == True:
    
    """Calculate the heat capacity for different temperatures"""
    heat_capacity = []
    temperatures = np.linspace(1, 4, num=100)
    for T in temperatures:
        total_energy = flipping(T, 0)[0]
        sigma_e = np.std(total_energy[2000:])
        heat_capacity.append(sigma_e**2 / (k_b * T**2))

    """Plot the heat capacity against temperature"""
    plt.plot(temperatures, heat_capacity)
    plt.xlabel("Temperature")
    plt.ylabel("Heat Capacity")

    """Polyfit line of best fit"""
    z = np.polyfit(temperatures, heat_capacity, 4)
    best_fit = np.poly1d(z)(temperatures)
    plt.plot(temperatures, best_fit)
    plt.show()

    """Determine the peak of the plot"""
    T_c = temperatures[np.argmax(best_fit)]
    print("Critical temperature = ", T_c)


"""Task 3"""
if task_3 == True:

    """Plot of finite-size scaling"""
    crit_temps = [2.545, 2.485, 2.455, 2.424, 2.424, 2.394, 2.394, 2.364, 2.394]
    N_vals = np.linspace(4, 20, num=9)
    plt.plot(N_vals, crit_temps, label="Raw data")
    plt.title("Finite Size Scaling")
    plt.xlabel("Grid size (N)")
    plt.ylabel("Critical Temperature")

    def finite_size_scaling(N, t_c_infinity, a, nu):
        return t_c_infinity + a * N ** (-1/nu)

    parameters, errors = fit(finite_size_scaling, N_vals, crit_temps)
    plt.plot(N_vals, finite_size_scaling(N_vals, parameters[0], parameters[1], parameters[2]), label="Best fit: t_c_infinity=%5.3f, a=%5.3f, nu=%5.3f" % tuple(parameters))
    plt.legend()
    plt.show()

    """Find the parameteres and their errors"""
    print(parameters)
    print(np.diag(errors)) #prints the variances of the parameters
    print(np.sqrt(np.diag(errors)))
        

"""Task 4"""
if task_4 == True:

    """Produce array of domain coordination numbers at different temperatures"""
    t4_temperatures = np.linspace(0.5, 10, num=10)
    domain_sizes = []
    for T in t4_temperatures:
        domain_sizes.append(rand_flipping(T, 0)[3])

    """Plot domain size against temperature"""
    plt.plot(t4_temperatures, domain_sizes)
    plt.axvline(x=2.27, c="black", ls="--", label="Critical Temperature")
    plt.title("Domain Sizes in 2D Ising Model")
    plt.xlabel("Temperature")
    plt.ylabel("Domain Size")
    plt.legend(loc="best")
    plt.show()

    """Produce image of model to observe domain size"""
    plt.imshow(rand_flipping(2.5, 0)[2])
    plt.title("Image of 2D Ising Model (T=2.5)")
    plt.show()


"""Task 5"""
if task_5 == True:

    """Cycle through H values and determine the average magnetisation"""
    hysteresis_m = []
    H_axis = np.linspace(-5, 5, num=400)
    for H in H_axis:
        rand_magnetism = rand_flipping(2, H)[1]
        average_hysteresis_m = np.average(rand_magnetism[10:])
        hysteresis_m.append(average_hysteresis_m)

    """Plot +ve and -ve values of rms magnetism against H to observe hysteresis"""
    plt.plot(H_axis, hysteresis_m, "b")
    #plt.plot(H_axis, savgol_filter(hysteresis_m, 9, 3), "r")
    plt.plot(H_axis, np.negative(hysteresis_m[::-1]), "b")
    #plt.plot(H_axis, savgol_filter(np.negative(hysteresis_m[::-1]), 9, 3), "r")
    plt.title("2D Ising Model")
    plt.xlabel("H")
    plt.ylabel("Magnetism")
    plt.show()




"""Stop time"""
print("Time taken by code:", time.clock() - start, "seconds")
