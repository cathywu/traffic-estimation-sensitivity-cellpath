from HighwayNetwork import HighwayNetwork
from random import randint, seed, random

########
# 
# Initialize randomizer
# 
########

import sys
#myseed = randint(0, sys.maxint)
myseed = 4909137950491786826
print "Random seed:", myseed
seed(myseed)

########
# 
# Load data
# 
########

import scipy.io
data = scipy.io.loadmat('sensitivity_sample.mat')
flows = [x[0] for x in data['x_true']]
routes = [map(tuple, x[0]) for x in data['paths_sampled']]

########
# 
# Initialize cell towers
# 
########

numCellTowers = 80

cellPositions = []
x = [s[0] for r in routes for s in r]
y = [s[1] for r in routes for s in r]
maxx = max(x)
maxy = max(y)
minx = min(x)
miny = min(y)
for i in xrange(numCellTowers):
  cellPositions.append((minx + random() * (maxx - minx), 
                        miny + random() * (maxy - miny)))

########
# 
# Pick a route
# 
########

index = 9
flows = [1]
routes = routes[index:index+1]

########
# 
# Plot stuff
# 
########

'''
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

vor = Voronoi(cellPositions)
voronoi_plot_2d(vor)
plt.plot(*zip(*routes[0]), marker='.')

for i, pt in enumerate(cellPositions):
  # Annotate the points 5 _points_ above and to the left of the vertex
  plt.annotate('{}'.format(i), xy=pt, xytext=(0, 0), ha='center', va='center',
              textcoords='offset points')

plt.show()
'''

########
# 
# Simulation
# 
########

# Run simulation
n = HighwayNetwork(cellPositions, flows, routes)

for f, rest, params in  n.go(2000, 10, tlimit = 100, spread=[0, 1], inertia=[0, 1], balancing=[0, .1]):
  print "spread = %f, inertia = %f, balancing = %f" % params
  for cp, count in n.paths.iteritems():
    print count, ":", cp
  #print f, rest
  print
