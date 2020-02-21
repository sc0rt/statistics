import datetime as dt
import numpy as np  
import pandas as pd  
import pandas_datareader as pdr  
import matplotlib.pyplot as plt  
from scipy.stats import norm

start = dt.datetime(2018,1,1)
end = dt.datetime.now()
ticker = 'INTC'
time_steps = 30 # the amount of steps taken into the future for the simulation
sims = 100 # the number of simulations that will be run

# Setting up data in a DataFrame
data = pd.DataFrame()
data[ticker] = pdr.DataReader(ticker, data_source='yahoo', start=start, end=end)['Adj Close']

# pct_change is the percentage change between current and prior element
log_returns = np.log(1 + data.pct_change())

# Setting up the values necessary for the daily_returns expression
mu = log_returns.mean() # average/mean
var = log_returns.var() # variance
drift = mu - (0.5 * var) # drift
sigma = log_returns.std() # standard deviation

daily_returns = np.exp(drift.values + sigma.values * norm.ppf(np.random.rand(time_steps, sims)))

# Takes last data point as as the starting point for the simulations
initial = data.iloc[-1]
price_list = np.zeros_like(daily_returns)
price_list[0] = initial

# For the amount of steps taken into the future, this loop calculated the simulation data to be plotted
# Takes input price, multiplies it by the exponential value calculated in daily_returns, sets simulated price
for t in range(1, time_steps):
    price_list[t] = price_list[t - 1] * daily_returns[t]

# Histogram for the price frequencies
plt.hist(price_list, bins=5, density=True)

# Fit a normal distribution to the price_list data
sim_mu, sim_sig = norm.fit(price_list)

# Probability Density Function
xmin, xmax = plt.xlim() # set the xmin and xmax along the axes for the pdf
x = np.linspace(xmin, xmax)
p = norm.pdf(x, sim_mu, sim_sig)

# Plots of raw data; logarithmic return; Monte Carlo Simulations; 
# and frequencies of the Monte Carle simulations fit to normal distribution
plt.plot(x, p, 'k')
data.plot(figsize=(12, 8))
log_returns.plot(figsize=(12, 8))
plt.figure(figsize=(12, 8))
plt.plot(price_list)
plt.show()
