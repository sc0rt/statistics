# Statistics For Engineers and Scientists, 4th Edition, Navidi – ISBN – 9781259275975
# STAT 2800 at UOIT
# CHAPTER 1: SAMPLING AND DESCRIPTIVE STATISTICS
# 1.3 Graphical Summaries

import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import stemgraphic

# Example 1: SAT scores copied from .pdf file
raw_sat = '638 574 627 621 705 690 522 612 594 581 640 653 638 760 491'
sat = []
for num in raw_sat.split(' '):
    sat.append(int(num)) # SAT scores now in a list as integer values

# Example 2: Stem-and-leaf plot of above data but two leaf categories per stem
# Can't seem to do this with stemgraphic. Only given the scale parameter

# Example 3: Student grades copied text from .pdf file
grade_data = '55 55 55 56 56 57 57 57 58 59 59 60 60 60 \
61 65 65 66 66 66 67 67 67 67 67 67 68 68 \
68 69 69 69 69 69 69 70 70 70 71 71 72 72 \
72 72 73 73 73 73 73 73 73 74 74 74 74 76 \
76 76 76 76 76 76 77 77 77 77 78 78 79 79 \
79 80 80 80 80 81 82 82 82 83 83 83 83 84 \
84 85 85 87 87 88 88 89 92 92 94'

grades = []
for num in grade_data.split(' '):
    grades.append(int(num)) # data now a list of integer grade values

# Histogram for grades
plt.hist(grades, bins=5, density=True) #desired bin size for normal distribution

# Fit a normal distribution to the grade data
mu, sigma = norm.fit(grades)

# Probability Density Function for grades
xmin, xmax = plt.xlim() #set the xmin and xmax along the axes for the pdf
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)

# Plot histogram with PDF
plt.plot(x, p, 'k')
plt.xlabel('Grade')
plt.ylabel('Relative Frequency')
title = "Historgram of Student Grades\nPDF fit results: mu = %.2f,  sigma = %.2f" % (mu, sigma)
plt.title(title)
plt.grid(True)

# Stem-and-Leaf Plot Example 1 SAT scores
fig2, ax2 = stemgraphic.stem_graphic(sat, scale=100, median_alpha=0)
fig2.suptitle("Stem-and-Leaf Plot of SAT Scores")

# Stem-and-Leaf Plot Example 3 Grades
fig3, ax3 = stemgraphic.stem_graphic(grades, scale=10, median_alpha=0)
fig3.suptitle("Stem-and-Leaf Plot of Student Grades")

# Box plot
fig4, ax4 = plt.subplots()
ax4.set_title('Box Plot for Student Grades')
ax4.boxplot(grades, vert=False)

plt.show()
