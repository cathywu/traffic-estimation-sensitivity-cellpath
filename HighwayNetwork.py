from random import shuffle, random
from math import log10
from copy import deepcopy

def uniq(seq):
    last = [None]
    setlast = lambda x: last.append(x)
    return tuple([ x for x in seq if not (x is None or x == last[-1] or setlast(x))])

class HighwayNetwork:

  def __init__(self, cellPositions, flows, routes, spread=0, inertia=0, balancing=0):
    self.spread = spread
    self.inertia = inertia
    self.balancing = balancing

    self.cellPositions = cellPositions
    cars = []
    for (flow, route) in zip(flows, routes):
      for i in xrange(flow):
        # TODO: disperse the cars in time by adding a random number of None's at the beginning of the route
        cars.append(route)
    #shuffle(cars)
    self.timesteps = map(list, map(None, *cars))

    self.buildRSSI()
    self.assignTowers()
    self.collect()

  def calculateRSSI(self, car, tower):
    # spread : random network interference degrading signal strength
    d2 = (car[0] - tower[0])**2 + (car[1] - tower[1])**2
    fade = 10*log10(d2) + self.spread*random()
    return -fade

  def buildRSSI(self):
    for ts in self.timesteps:
      for i in xrange(len(ts)):
        if ts[i] is None:
          continue
        rssis = []
        for cell in self.cellPositions:
          rssis.append(self.calculateRSSI(ts[i], cell))
        ts[i] = rssis

  def assignTowers(self):
    for i, ts in enumerate(self.timesteps):
      towerLoad = [0] * len(self.cellPositions)
      for c, car in enumerate(ts):
        # inertia : stay on the same tower
        if car is None:
          continue
        if i > 0:
          lastTower = self.timesteps[i-1][c]
          car[lastTower] += self.inertia

        # balancing : penalize heavily loaded towers
        for t, load in enumerate(towerLoad):
          car[t] -= self.balancing * load

        # pick winner
        tower = car.index(max(car))
        self.timesteps[i][c] = tower
        towerLoad[tower] += 1

  def collect(self):
    cars = map(uniq, map(None, *self.timesteps))
    paths = {}
    for car in cars:
      paths[car] = 1 + paths.get(car, 0)
    self.paths = paths
   
  def toString(self):
    for ts in self.timesteps:
      for car in ts:
        print car
      print
    print "~~~"

if __name__ == "__main__": 
  simple = True

  if simple:
    cellPositions = [ (1,1), (1,3), (3,1) ]
    flows = [ 1, 2, 3, 4 ]
    routes = [ [ (t, 2*t) for t in range(3) ], 
               [ (2*t, t) for t in range(3) ], 
               [ (t, 3-t) for t in range(4) ], 
               [ (2*t, 2*t) for t in range(3) ], 
             ]
  else:
    import scipy.io
    data = scipy.io.loadmat('sensitivity_sample.mat')
    data['paths_sampled'][N][0]


  n = HighwayNetwork(cellPositions, flows, routes)
  print n.paths
