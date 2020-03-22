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


