# https://www.lcbo.com/content/dam/lcbo/corporate-pages/about/pdf/LCBO_AR17-18-english.pdf

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

time_steps = 6 # 5 Years into the future
sims = 30 # 30 simulations for each time step

# Net income in billions of dollars
data = pd.DataFrame({'Fiscal Year' : [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
                    'Expenses to Sales Ratio' : [1.37, 1.41, 1.44, 1.56, 1.66, 1.71, 1.74, 1.82, 1.97, 2.07, 2.21]})
data.set_index('Fiscal Year', inplace=True)

# pct_change is the percentage change between current and prior element
log_ni = np.log(1 + data.pct_change())

# Setting up the values necessary for the yearly_ni expression
mu = log_ni.mean() # average/mean
var = log_ni.var() # variance
drift = mu - (0.5 * var) # drift
sigma = log_ni.std() # standard deviation

# ers is expenses to sales ratio
yearly_ni = np.exp(drift.values + sigma.values * norm.ppf(np.random.rand(time_steps, sims)))

# Takes last data point as as the starting point for the simulations
initial = data.iloc[-1]
monte_list = np.zeros_like(yearly_ni)
monte_list[0] = initial

# For the amount of steps taken into the future, this loop calculated the simulation data to be plotted
# Takes input price, multiplies it by the exponential value calculated in yearly_ni, sets simulated price
for t in range(1, time_steps):
    monte_list[t] = monte_list[t - 1] * yearly_ni[t]

# Histogram for the price frequencies, number of bins can be adjusted
plt.figure(figsize=(10, 6))
plt.hist(monte_list[1], bins=5, density=True)

# Fit a normal distribution to the first simulation point in monte_list data
# Could implement for each successive iteration in the monte_list index (monte_list[2], monte_list[3], ..., monte_list[time_steps - 1])
sim_mu, sim_sig = norm.fit(monte_list[1])

# Probability Density Function
xmin, xmax = plt.xlim() # set the xmin and xmax along the x-axis for the pdf
x = np.linspace(xmin, xmax)
p = norm.pdf(x, sim_mu, sim_sig)

# Plot Histogram with probability density function
plt.plot(x, p, 'k') # normal distribution fit
plt.xlabel('Expenses to Sales Ratio')
plt.ylabel('Probability Density')
title = "Histogram for 30 Simulations for Net Income 1 Year into the Future\nPDF fit results: mu = %.4f,  sigma = %.4f" % (sim_mu, sim_sig)
plt.title(title)

# Plot of 30 Monte Carlo Simulations for each year into the future
plt.figure(figsize=(10, 6))
plt.plot(monte_list) # monte carlo
plt.xlabel('Years into the Future')
plt.ylabel('Net Income in Billions of Dollars')
title = "Net Income Monte Carlo Simulations (n = 30)"
plt.title(title)
plt.show()
