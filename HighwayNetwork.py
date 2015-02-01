from __future__ import division
import ipdb
from random import shuffle, random, uniform, randint, seed
from math import log10
from itertools import product
import logging

def uniq(seq):
    last = [None]
    setlast = lambda x: last.append(x)
    return tuple([ x for x in seq if not (x is None or x == last[-1] or setlast(x))])

def weighted_choice(weights):
   total = sum(weights)
   r = uniform(0, total)
   upto = 0
   for i, w in enumerate(weights):
      upto += w
      if upto > r:
         return i
   assert False, "Shouldn't get here"

class HighwayNetwork:

  def __init__(self, cellPositions, flows, routes, threshold = 1e-4):
    self.cellPositions = cellPositions
    self.flows = [x for x in flows if x > threshold]
    self.routes = [x for (i, x) in enumerate(routes) if flows[i] > threshold]
    logging.info( "Calculating signal strength...")
    self.buildRSSI()

  def go(self, numCars, numDelays, tlimit=None, spread=None, inertia=None, balancing=None, cellpaths=None):
    if spread is None:
      spread = [0]
    elif not isinstance(spread, (list, tuple)):
      spread = [spread]

    if inertia is None:
      inertia = [0]
    elif not isinstance(inertia, (list, tuple)):
      inertia = [inertia]

    if balancing is None:
      balancing = [0]
    elif not isinstance(balancing, (list, tuple)):
      balancing = [balancing]

    logging.info( "Deploying cars...")
    self.deployCars(numCars, numDelays, tlimit)
    for s in spread:
      logging.info( "Adding RF noise (%f) ..." % s)
      self.addNoise(s)
      for i, b in product(inertia, balancing):
        logging.info( "Determining tower assignment (inertia=%f, balancing=%f) ..." % (i, b))
        self.assignTowers(i, b)
        logging.info( "Done.")
        f, rest = self.collect(numCars * numDelays, cellpaths = cellpaths)
        yield f, rest, (s, i, b)

  def deployCars(self, numCars, numDelays, tlimit=None):
    cars = []
    for d in xrange(numDelays):
      # add numCars every timestep for numDelay timesteps
      delay = [None] * d
      for c in xrange(numCars):
        # pick a route for each car according to flows
        route = weighted_choice(self.flows)
        cars.append(delay + self.routes[route][:])
    shuffle(cars)
    # transpose to get where each car is at each timestep
    self.timesteps = map(list, map(None, *cars))
    if tlimit:
      self.timesteps = self.timesteps[:tlimit]

  def addNoise(self, spread=0):
    # spread : random network interference degrading signal strength
    self.noisy = [[None for x in ts] for ts in self.timesteps]
    for t, ts in enumerate(self.timesteps):
      for c, car in enumerate(ts):
        if car is None:
          continue
        self.noisy[t][c] = [x - spread*random() for x in car]

  def calculateRSSI(self, car, tower):
    # spread : random network interference degrading signal strength
    d2 = (car[0] - tower[0])**2 + (car[1] - tower[1])**2
    fade = 10*log10(d2)
    return -fade

  def buildRSSI(self):
    for route in self.routes:
      for i in xrange(len(route)):
        if route[i] is None:
          continue
        rssis = []
        for cell in self.cellPositions:
          rssis.append(self.calculateRSSI(route[i], cell))
        route[i] = rssis

  def assignTowers(self, inertia, balancing):
    # inertia : preference for an individual car to stay on the same tower
    # balancing : disincentivize more heavily loaded towers
    self.towers = [[None for x in ts] for ts in self.noisy]
    towerLoad = [0] * len(self.cellPositions)
    for i, ts in enumerate(self.noisy):
      for c, car in enumerate(ts):
        # inertia : stay on the same tower
        if car is None:
          continue
        towers = car[:]
        if i > 0:
          lastTower = self.towers[i-1][c]
          if lastTower is not None:
            towers[lastTower] += inertia
            towerLoad[lastTower] -= 1

        # balancing : penalize heavily loaded towers
        for t, load in enumerate(towerLoad):
          towers[t] -= balancing * load

        # pick winner
        tower = towers.index(max(towers))
        self.towers[i][c] = tower
        towerLoad[tower] += 1

  def getCellpaths(self):
    return self.paths.keys()

  def collect(self, total, cellpaths=None):
    cars = map(uniq, map(None, *self.towers))
    paths = {}
    for car in cars:
      paths[car] = 1 + paths.get(car, 0)
    self.paths = paths

    if cellpaths:
      f = [0] * len(cellpaths)
      for i, cp in enumerate(cellpaths):
        if cp in paths:
          f[i] = paths[cp] * sum(self.flows) / total
    else:
      f = []
    return f, sum(self.flows) - sum(f)

if __name__ == "__main__": 
  import sys
  myseed = randint(0, sys.maxint)
  myseed = 4909137950491786826
  print "Random seed:", myseed
  seed(myseed)

  simple = False

  """
  Set up environment
  """
  if simple:
    cellPositions = [ (1,1), (1,3), (3,1) ]
    flows = [ 1, 2, 3, 4 ]
    routes = [ [ (t, 2*t) for t in range(3) ], 
               [ (2*t, t) for t in range(3) ], 
               [ (t, 3-t) for t in range(4) ], 
               [ (2*t, 2*t) for t in range(3) ], 
             ]
  else:
    numCellTowers = 80

    import scipy.io
    data = scipy.io.loadmat('sensitivity_sample.mat')
    flows = [x[0] for x in data['x_true']]
    routes = [map(tuple, x[0]) for x in data['paths_sampled']]

    cellPositions = []
    x = [s[0] for r in routes for s in r]
    y = [s[1] for r in routes for s in r]
    maxx = max(x)
    maxy = max(y)
    minx = min(x)
    miny = min(y)
    for i in xrange(numCellTowers):
      cellPositions.append((minx + random() * (maxx - minx), miny + random() * \
                            (maxy - miny)))

  """
  Simulation
  """
  # Run simulation
  n = HighwayNetwork(cellPositions, flows, routes)
  for f, rest, params in  n.go(200, 10, tlimit = 100, inertia=[0, 0.02, 0.5], balancing = [0, .02, .05], cellpaths=[ (34, 20, 67, 40, 42), (77, 74, 67, 42, 36, 42), (54, 72, 40, 67, 42, 36) ]):
    print "spread = %f, inertia = %f, balancing = %f" % params
    print f, rest
    print
  '''
  # Print results
  for k, v in n.paths.iteritems():
    print k, ":", v
  print len(n.paths)
  '''
