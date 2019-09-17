#Ising Setup
#Kieran Hobden
#31-Mar-'19

"""Create an nxn grid"""
n = 10
all_up_grid = np.ones((n, n))

"""Initiate random number generator"""
random.seed(30)

"""Define constants"""
J = 1
mu = 1
H = 0
k_b = 1

"""Define number of times to analyse flip of every component"""
N = 50

"""Compute the sum over neighbours for use in the energy calculation and to
increase efficiency """
def neighbour_sum(i, j):
    sum = grid[i][j]*(grid[(i+1)%n][j%n] + grid[(i-1)%n][j%n] + \
    grid[i%n][(j+1)%n] + grid[i%n][(j-1)%n])
    return sum

"""Compute magnetism"""
def magnetism():
    magnetism = np.sum(grid)

"""Compute the energy of the grid"""
def energy():
    int_term = 0.0
    for i in range(n):
        for j in range(n):
            int_term += neighbour_sum(i, j)
    spin_sum = magnetism()
    energy = -(J/2)*int_term - mu*H*spin_sum
    return energy

"""Flipping algorithm"""
x_val = np.arange(N*n**2)
def flipping(T):
    y_val = np.zeros(N*n**2)
    global grid
    grid = np.ones((n, n))
    for k in range(N):
        for i in range(n):
            for j in range(n):
                e_a = energy()
                y_val[k*n**2 + i*n + j] = e_a
                grid[i][j] = -grid[i][j]
                e_b = e_a + 2*neighbour_sum(i, j)
                if e_b > e_a and random.random() < math.exp((e_a - e_b)/(k_b*T)):
                    grid[i][j] = -grid[i][j]
    return y_val

"""Stop time"""
print("Time taken by code:", time.clock() - start, "seconds")
