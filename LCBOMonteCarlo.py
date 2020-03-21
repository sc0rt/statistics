import datetime as dt
import numpy as np 
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from scipy.stats import norm

time_steps = 5 # 5 Years into the future
sims = 20 # 20 simulations for each time step

data = pd.DataFrame({'Fiscal Year' : [2014, 2015, 2016, 2017, 2018],
                    'Expenses to Sales Ratio' : [0.163, 0.160, 0.156, 0.160, 0.158]})
data.set_index('Fiscal Year', inplace=True)

# pct_change is the percentage change between current and prior element
log_esr = np.log(1 + data.pct_change())

# Setting up the values necessary for the yearly_esr expression
mu = log_esr.mean() # average/mean
var = log_esr.var() # variance
drift = mu - (0.5 * var) # drift
sigma = log_esr.std() # standard deviation

# ers is expenses to sales ratio
yearly_esr = np.exp(drift.values + sigma.values * norm.ppf(np.random.rand(time_steps, sims)))

# Takes last data point as as the starting point for the simulations
initial = data.iloc[-1]
monte_list = np.zeros_like(yearly_esr)
monte_list[0] = initial

# For the amount of steps taken into the future, this loop calculated the simulation data to be plotted
# Takes input price, multiplies it by the exponential value calculated in yearly_esr, sets simulated price
for t in range(1, time_steps):
    monte_list[t] = monte_list[t - 1] * yearly_esr[t]

# Histogram for the price frequencies, number of bins can be adjusted
plt.figure(figsize=(10, 6))
plt.hist(monte_list[1], bins=3, density=True)

# Fit a normal distribution to the first simulation point in monte_list data
# Could implement for each successive iteration in the monte_list index (monte_list[2], monte_list[3], ..., monte_list[time_steps - 1])
sim_mu, sim_sig = norm.fit(monte_list[1])

# Probability Density Function
xmin, xmax = plt.xlim() # set the xmin and xmax along the x-axis for the pdf
x = np.linspace(xmin, xmax)
p = norm.pdf(x, sim_mu, sim_sig)

# Plot for the first simulation fit to normal distribution, and plot for Monte Carlo Simulation
plt.plot(x, p, 'k') # normal distribution fit
plt.figure(figsize=(10, 6))
plt.plot(monte_list) # monte carlo
plt.show()
